// Timing bench: wav2vec2 CTC (base / large) on-device via ONNX Runtime 1.20 (CPU EP and Core ML EP), fixed 6 s input, then CTC candidate scoring.
// Public models (Apache-2.0), public-domain clip. Only time and memory matter; argmax agreement with the PyTorch reference is a sanity check.
import SwiftUI
import Foundation
import OnnxRuntimeBindings

func nowMs() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }

func footprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
    }
    return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1048576 : -1
}

func thermal() -> Int { ProcessInfo.processInfo.thermalState.rawValue }

// log P(target | frames) with CTC (blank = 0); lp is T x V log-softmax, row-major.
func ctcLogProb(_ lp: [Float], T: Int, V: Int, target: [Int]) -> Double {
    let L = target.count, S = 2 * L + 1
    let neg = -Double.infinity
    func lse(_ a: Double, _ b: Double) -> Double { if a == neg { return b }; if b == neg { return a }; let m = max(a, b); return m + log(exp(a - m) + exp(b - m)) }
    var prev = [Double](repeating: neg, count: S), cur = prev
    func sym(_ s: Int) -> Int { s % 2 == 0 ? 0 : target[s / 2] }
    prev[0] = Double(lp[0]); if S > 1 { prev[1] = Double(lp[target[0]]) }
    for t in 1..<T {
        for s in 0..<S {
            var a = prev[s]
            if s > 0 { a = lse(a, prev[s - 1]) }
            if s > 1 && sym(s) != 0 && sym(s) != sym(s - 2) { a = lse(a, prev[s - 2]) }
            cur[s] = a == neg ? neg : a + Double(lp[t * V + sym(s)])
        }
        swap(&prev, &cur)
    }
    return S > 1 ? lse(prev[S - 1], prev[S - 2]) : prev[0]
}

func logSoftmax(_ x: [Float], T: Int, V: Int) -> [Float] {
    var out = x
    for t in 0..<T {
        var m = -Float.infinity
        for v in 0..<V { m = max(m, x[t * V + v]) }
        var s: Float = 0
        for v in 0..<V { s += exp(x[t * V + v] - m) }
        let l = log(s)
        for v in 0..<V { out[t * V + v] = x[t * V + v] - m - l }
    }
    return out
}

struct Ref: Decodable { let ids: [Int]; let T: Int; let V: Int }

@MainActor final class Model: ObservableObject {
    @Published var rows: [String] = ["starting"]
    var started = false
    func add(_ s: String) { rows.append(s); print("BENCH " + s) }

    func start() {
        if started { return }
        started = true
        Task.detached(priority: .userInitiated) { [self] in
            let b = Bundle.main
            let vocab = try! JSONDecoder().decode([String: Int].self, from: Data(contentsOf: b.url(forResource: "vocab", withExtension: "json")!))
            let refs = try! JSONDecoder().decode([String: Ref].self, from: Data(contentsOf: b.url(forResource: "ref", withExtension: "json")!))
            let clipData = try! Data(contentsOf: b.url(forResource: "clip", withExtension: "f32")!)
            let n = clipData.count / 4
            await add("INFO cpus=\(ProcessInfo.processInfo.processorCount) ram=\(ProcessInfo.processInfo.physicalMemory / 1048576)MB samples=\(n) base=\(Int(footprintMB()))MB")
            let env = try! ORTEnv(loggingLevel: .warning)
            // order: cheap first so rows survive a later memory kill
            let plan: [(String, String)] = [("base", "cpu"), ("large", "cpu"), ("base", "coreml-all"), ("base", "coreml-gpu"), ("large", "coreml-all")]
            for (name, ep) in plan {
                var peak = 0.0
                do {
                    let path = b.path(forResource: "w2v-\(name)", ofType: "onnx")!
                    let t0 = nowMs()
                    let so = try ORTSessionOptions()
                    try so.setIntraOpNumThreads(4)
                    if ep != "cpu" {
                        let co = ORTCoreMLExecutionProviderOptions()
                        co.createMLProgram = true
                        if ep == "coreml-gpu" { co.useCPUAndGPU = true }
                        try so.appendCoreMLExecutionProvider(with: co)
                    }
                    let session = try ORTSession(env: env, modelPath: path, sessionOptions: so)
                    await add("LOAD \(name) \(ep) ms=\(Int(nowMs() - t0)) foot=\(Int(footprintMB()))MB")
                    peak = max(peak, footprintMB())
                    var lastLogits = [Float](); var T = 0, V = 0
                    for run in 0..<5 {
                        let data = NSMutableData(data: clipData)
                        let input = try ORTValue(tensorData: data, elementType: .float, shape: [1, NSNumber(value: n)])
                        let t1 = nowMs()
                        let outs = try session.run(withInputs: ["input_values": input], outputNames: ["logits"], runOptions: nil)
                        let ms = nowMs() - t1
                        let out = outs["logits"]!
                        let shape = try out.tensorTypeAndShapeInfo().shape.map { $0.intValue }
                        T = shape[1]; V = shape[2]
                        let td = try out.tensorData() as Data
                        lastLogits = td.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                        peak = max(peak, footprintMB())
                        await add("R|\(name)|\(ep)|run\(run)\(run == 0 ? "(cold)" : "")|ms=\(Int(ms))|foot=\(Int(footprintMB()))|peak=\(Int(peak))|th=\(thermal())")
                    }
                    // sanity: argmax agreement with the PyTorch reference
                    var agree = 0
                    for t in 0..<T { var bi = 0; for v in 1..<V where lastLogits[t * V + v] > lastLogits[t * V + bi] { bi = v }; if let r = refs[name], t < r.ids.count, r.ids[t] == bi { agree += 1 } }
                    await add("ACC|\(name)|\(ep)|argmax_agree=\(agree)/\(T)")
                    // CTC candidate scoring: 260 candidate strings (the heavy-route size) scored against the one pass of logits
                    let lp = logSoftmax(lastLogits, T: T, V: V)
                    let words = ["AND", "SO", "MY", "FELLOW", "AMERICANS", "ASK", "NOT", "WHAT", "YOUR", "COUNTRY", "CAN", "DO", "FOR", "YOU"]
                    var cands: [[Int]] = []
                    for i in 0..<260 {
                        var s = ""
                        for k in 0..<5 { s += (k > 0 ? "|" : "") + words[(i * 7 + k * 3) % words.count] }
                        cands.append(s.map { vocab[String($0)] ?? 3 })
                    }
                    let d0 = nowMs()
                    var best = -Double.infinity
                    for c in cands { best = max(best, ctcLogProb(lp, T: T, V: V, target: c)) }
                    await add("DP|\(name)|\(ep)|cands=260|len~\(cands[0].count)|ms=\(Int(nowMs() - d0))|best=\(String(format: "%.1f", best))")
                } catch {
                    await add("ERR \(name) \(ep) \(String(describing: error).prefix(160))")
                }
            }
            await add("DONE")
        }
    }
}

@main struct W2VBenchApp: App {
    @StateObject var m = Model()
    var body: some Scene {
        WindowGroup {
            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(m.rows.enumerated()), id: \.offset) { item in
                        Text(item.element).font(.system(size: 9, design: .monospaced)).accessibilityLabel(item.element)
                    }
                }.padding(6)
            }.onAppear { m.start() }
        }
    }
}
