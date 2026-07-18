import Foundation
import CoreML

setvbuf(stdout, nil, _IOLBF, 1024)

// Использование: swift decode_bench.swift <compute_units: all|ane|cpu|cpugpu> <n_iters> <model: 0.4b|1.5b>
let args = CommandLine.arguments
let cuArg = args.count > 1 ? args[1] : "ane"
let nIters = args.count > 2 ? Int(args[2])! : 100
let modelArg = args.count > 3 ? args[3] : "1.5b"

func computeUnits(_ s: String) -> MLComputeUnits {
    switch s {
    case "all": return .all
    case "cpu": return .cpuOnly
    case "cpugpu": return .cpuAndGPU
    default: return .cpuAndNeuralEngine
    }
}

let cfg = MLModelConfiguration()
cfg.computeUnits = computeUnits(cuArg)

func loadModel(_ path: String) throws -> MLModel {
    let expanded = (path as NSString).expandingTildeInPath
    let url = URL(fileURLWithPath: expanded)
    return try MLModel(contentsOf: url, configuration: cfg)
}

func mkInt32(_ v: Int32) throws -> MLMultiArray {
    let a = try MLMultiArray(shape: [1, 1], dataType: .int32)
    a[0] = NSNumber(value: v)
    return a
}

func argmax65536(_ arr: MLMultiArray) -> Int32 {
    var best: Int32 = 0
    var bestV: Float = -1e30
    let ptr = arr.dataPointer.bindMemory(to: Float16.self, capacity: 65536)
    for i in 0..<65536 {
        let v = Float(ptr[i])
        if v > bestV { bestV = v; best = Int32(i) }
    }
    return best
}

print("compute_units=\(cuArg) iters=\(nIters) model=\(modelArg)")

let t_load0 = Date()

if modelArg == "0.4b" {
    let m = try loadModel("~/Develop/rwkv7-g1d-0.4b-20260210-ctx8192-coreml-lut8/rwkv7-g1d-0.4b-ctx8192-coreml-lut8_chunk1of1.mlmodelc")
    print("load: \(Date().timeIntervalSince(t_load0))s")
    let state = m.makeState()
    var tok: Int32 = 123
    func step() throws {
        let inp = try MLDictionaryFeatureProvider(dictionary: ["in0": mkInt32(tok)])
        let out = try m.prediction(from: inp, using: state)
        let logits = out.featureValue(for: "out0")!.multiArrayValue!
        tok = argmax65536(logits)
    }
    for _ in 0..<10 { try step() }
    let t0 = Date()
    for _ in 0..<nIters { try step() }
    let dt = Date().timeIntervalSince(t0)
    print("DONE: \(nIters) iters, \(dt/Double(nIters)*1000) ms/tok, \(Double(nIters)/dt) tok/s")
} else {
    let m1 = try loadModel("~/Develop/rwkv7-g1c-1.5b-20260110-ctx8192-coreml-lut6/rwkv7-g1c-1.5b-ctx8192_combined_lut6_chunk1of2.mlmodelc")
    let m2 = try loadModel("~/Develop/rwkv7-g1c-1.5b-20260110-ctx8192-coreml-lut6/rwkv7-g1c-1.5b-ctx8192_combined_lut6_chunk2of2.mlmodelc")
    print("load: \(Date().timeIntervalSince(t_load0))s")
    let s1 = m1.makeState()
    let s2 = m2.makeState()
    var tok: Int32 = 123
    func step() throws {
        let inp1 = try MLDictionaryFeatureProvider(dictionary: ["in0": mkInt32(tok)])
        let out1 = try m1.prediction(from: inp1, using: s1)
        let hidden = out1.featureValue(for: "out0")!.multiArrayValue!
        let vfirst = out1.featureValue(for: "v_first_out")!.multiArrayValue!
        let inp2 = try MLDictionaryFeatureProvider(dictionary: [
            "in0": MLFeatureValue(multiArray: hidden),
            "v_first_in": MLFeatureValue(multiArray: vfirst)
        ])
        let out2 = try m2.prediction(from: inp2, using: s2)
        let logits = out2.featureValue(for: "out0")!.multiArrayValue!
        tok = argmax65536(logits)
    }
    for _ in 0..<10 { try step() }
    let t0 = Date()
    for _ in 0..<nIters { try step() }
    let dt = Date().timeIntervalSince(t0)
    print("DONE: \(nIters) iters, \(dt/Double(nIters)*1000) ms/tok, \(Double(nIters)/dt) tok/s")
}
