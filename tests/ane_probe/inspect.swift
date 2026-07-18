import Foundation
import CoreML

let path = CommandLine.arguments[1]
let url = URL(fileURLWithPath: path)

do {
    let model = try MLModel(contentsOf: url)
    let desc = model.modelDescription
    print("=== INPUTS ===")
    for (name, d) in desc.inputDescriptionsByName {
        print("\(name): \(d)")
    }
    print("=== OUTPUTS ===")
    for (name, d) in desc.outputDescriptionsByName {
        print("\(name): \(d)")
    }
    print("=== STATE (if any) ===")
    if #available(macOS 15.0, *) {
        for (name, d) in desc.stateDescriptionsByName {
            print("\(name): \(d)")
        }
    }
    print("=== METADATA ===")
    for (k, v) in desc.metadata {
        print("\(k): \(v)")
    }
} catch {
    print("ERROR: \(error)")
}
