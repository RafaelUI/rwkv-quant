import Foundation
import CoreML

setvbuf(stdout, nil, _IOLBF, 1024)

let path = "~/Develop/rwkv7-g1d-0.4b-20260210-ctx8192-coreml-lut8/rwkv7-g1d-0.4b-ctx8192-coreml-lut8_chunk1of1.mlmodelc"
let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)

func run() async {
    do {
        let asset = try MLModelAsset(url: url)
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine
        cfg.functionName = "decode"
        let m = try await MLModel.load(asset: asset, configuration: cfg)
        print("LOADED OK via MLModelAsset, functionName=decode")
        print(m.modelDescription.inputDescriptionsByName)
    } catch {
        print("ASSET ERROR: \(error)")
    }
}

let sem = DispatchSemaphore(value: 0)
Task {
    await run()
    sem.signal()
}
sem.wait()
