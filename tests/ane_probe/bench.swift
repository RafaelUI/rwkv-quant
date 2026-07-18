import Foundation
import CoreML

setvbuf(stdout, nil, _IOLBF, 1024)

// swift bench.swift <compute: all|ane|cpu|cpugpu> <mode: decode|prefill> <model: 0.4b|1.5b> <T tokens total>
let args = CommandLine.arguments
let cuArg = args.count > 1 ? args[1] : "ane"
let mode = args.count > 2 ? args[2] : "decode"
let modelArg = args.count > 3 ? args[3] : "1.5b"
let T = args.count > 4 ? Int(args[4])! : (mode == "decode" ? 40 : 1024)

func computeUnits(_ s: String) -> MLComputeUnits {
    switch s {
    case "all": return .all
    case "cpu": return .cpuOnly
    case "cpugpu": return .cpuAndGPU
    default: return .cpuAndNeuralEngine
    }
}

func loadModel(_ path: String, function: String) throws -> MLModel {
    let expanded = (path as NSString).expandingTildeInPath
    let url = URL(fileURLWithPath: expanded)
    let cfg = MLModelConfiguration()
    cfg.computeUnits = computeUnits(cuArg)
    cfg.functionName = function
    return try MLModel(contentsOf: url, configuration: cfg)
}

func mkInt32Chunk(_ n: Int, start: Int32) throws -> MLMultiArray {
    let a = try MLMultiArray(shape: [1, NSNumber(value: n)], dataType: .int32)
    for i in 0..<n { a[i] = NSNumber(value: start + Int32(i)) }
    return a
}

func argmaxLast(_ arr: MLMultiArray, vocab: Int = 65536) -> Int32 {
    var best: Int32 = 0
    var bestV: Float = -1e30
    let ptr = arr.dataPointer.bindMemory(to: Float16.self, capacity: vocab)
    for i in 0..<vocab {
        let v = Float(ptr[i])
        if v > bestV { bestV = v; best = Int32(i) }
    }
    return best
}

print("compute=\(cuArg) mode=\(mode) model=\(modelArg) T=\(T)")
let t_load0 = Date()

let base04 = "~/Develop/rwkv7-g1d-0.4b-20260210-ctx8192-coreml-lut8/rwkv7-g1d-0.4b-ctx8192-coreml-lut8_chunk1of1.mlmodelc"
let base15c1 = "~/Develop/rwkv7-g1c-1.5b-20260110-ctx8192-coreml-lut6/rwkv7-g1c-1.5b-ctx8192_combined_lut6_chunk1of2.mlmodelc"
let base15c2 = "~/Develop/rwkv7-g1c-1.5b-20260110-ctx8192-coreml-lut6/rwkv7-g1c-1.5b-ctx8192_combined_lut6_chunk2of2.mlmodelc"

if modelArg == "0.4b" {
    let chunkLen = (mode == "prefill") ? 16 : 1
    let m = try loadModel(base04, function: mode)
    print("load: \(Date().timeIntervalSince(t_load0))s")
    let state = m.makeState()
    var tok: Int32 = 100

    func step() throws {
        let inp = try MLDictionaryFeatureProvider(dictionary: ["in0": mkInt32Chunk(chunkLen, start: tok)])
        let out = try m.prediction(from: inp, using: state)
        let logits = out.featureValue(for: "out0")!.multiArrayValue!
        tok = argmaxLast(logits)
    }
    let nSteps = max(1, T / chunkLen)
    for _ in 0..<3 { try step() }
    let t0 = Date()
    for _ in 0..<nSteps { try step() }
    let dt = Date().timeIntervalSince(t0)
    let totalTok = nSteps * chunkLen
    print("DONE: \(nSteps) вызовов x \(chunkLen) ток = \(totalTok) ток, \(dt/Double(totalTok)*1000) ms/ток, \(Double(totalTok)/dt) ток/с")
} else {
    let chunkLen = (mode == "prefill") ? 32 : 1
    let m1 = try loadModel(base15c1, function: mode)
    let m2 = try loadModel(base15c2, function: mode)
    print("load: \(Date().timeIntervalSince(t_load0))s")
    let s1 = m1.makeState()
    let s2 = m2.makeState()
    var tok: Int32 = 100

    func step() throws {
        let inp1 = try MLDictionaryFeatureProvider(dictionary: ["in0": mkInt32Chunk(chunkLen, start: tok)])
        let out1 = try m1.prediction(from: inp1, using: s1)
        let hidden = out1.featureValue(for: "out0")!.multiArrayValue!
        let vfirst = out1.featureValue(for: "v_first_out")!.multiArrayValue!
        let inp2 = try MLDictionaryFeatureProvider(dictionary: [
            "in0": MLFeatureValue(multiArray: hidden),
            "v_first_in": MLFeatureValue(multiArray: vfirst)
        ])
        let out2 = try m2.prediction(from: inp2, using: s2)
        let logits = out2.featureValue(for: "out0")!.multiArrayValue!
        tok = argmaxLast(logits)
    }
    let nSteps = max(1, T / chunkLen)
    for _ in 0..<3 { try step() }
    let t0 = Date()
    for _ in 0..<nSteps { try step() }
    let dt = Date().timeIntervalSince(t0)
    let totalTok = nSteps * chunkLen
    print("DONE: \(nSteps) вызовов x \(chunkLen) ток = \(totalTok) ток, \(dt/Double(totalTok)*1000) ms/ток, \(Double(totalTok)/dt) ток/с")
}
