import MetalPerformanceShadersGraph

protocol PackedBoardSequence {
    var width: Int { get }
    var points: [Int] { get }
    func gather(_ graph: MPSGraph, _ source: MPSGraphTensor) -> MPSGraphTensor
    func restore(_ graph: MPSGraph, _ source: MPSGraphTensor) -> MPSGraphTensor
}

extension YTriangleSequence: PackedBoardSequence {}

// Each row packs its own board into the largest supported token count.
// Gather/scatter maps and attention bias are selected by the row's board mask.
struct BentYSequence: PackedBoardSequence {
    let width: Int
    let points: [Int]
    let gatherMap: MPSGraphTensor
    let restoreMap: MPSGraphTensor

    static func indices(_ graph: MPSGraph, _ values: [Int]) -> MPSGraphTensor {
        let codes = values.map { Int32($0) }
        let data = codes.withUnsafeBufferPointer { Data(buffer: $0) }
        return graph.constant(data, shape: [1, codes.count as NSNumber], dataType: .int32)
    }

    static func canvasPoints(_ geometry: GraphAttentionGeometry, size: Int, width: Int) -> [Int] {
        geometry.points.map { ($0 / size) * width + $0 % size }
    }

    static func make(
        graph: MPSGraph, width: Int,
        geometries: [Int: GraphAttentionGeometry], mask: MPSGraphTensor
    ) -> any PackedBoardSequence {
        let sizes = geometries.keys.filter { $0 <= width }.sorted()
        let size = sizes.last!
        if sizes.count == 1 {
            return YTriangleSequence(width: width,
                points: canvasPoints(geometries[size]!, size: size, width: width))
        }
        return BentYSequence(graph: graph, width: width, geometries: geometries, mask: mask)
    }

    init(graph: MPSGraph, width: Int, geometries: [Int: GraphAttentionGeometry], mask: MPSGraphTensor) {
        self.width = width
        let sizes = geometries.keys.filter { $0 <= width }.sorted(by: >)
        points = Self.canvasPoints(geometries[sizes[0]]!, size: sizes[0], width: width)
        let tokens = points.count
        let count = graph.reshape(graph.reductionSum(with: mask, axes: [1, 2, 3], name: nil),
            shape: [-1, 1], name: nil)
        var gather: MPSGraphTensor?
        var restore: MPSGraphTensor?
        for size in sizes {
            let playable = Self.canvasPoints(geometries[size]!, size: size, width: width)
            let forward = playable + [Int](repeating: width * width, count: tokens - playable.count)
            var reverse = [Int](repeating: tokens, count: width * width)
            for (index, point) in playable.enumerated() { reverse[point] = index }
            let forwardTensor = Self.indices(graph, forward)
            let reverseTensor = Self.indices(graph, reverse)
            if let currentGather = gather, let currentRestore = restore {
                let matches = graph.equal(count,
                    graph.constant(Double(playable.count), dataType: .float32), name: nil)
                gather = graph.select(predicate: matches, trueTensor: forwardTensor, falseTensor: currentGather, name: nil)
                restore = graph.select(predicate: matches, trueTensor: reverseTensor, falseTensor: currentRestore, name: nil)
            } else {
                gather = forwardTensor
                restore = reverseTensor
            }
        }
        gatherMap = gather!
        restoreMap = restore!
    }

    func remap(_ graph: MPSGraph, _ source: MPSGraphTensor, map: MPSGraphTensor,
               sourcePoints: Int, targetHeight: Int, targetWidth: Int) -> MPSGraphTensor {
        let channels = source.shape![1]
        let flat = graph.reshape(source, shape: [-1, channels, sourcePoints as NSNumber], name: nil)
        let first = graph.sliceTensor(flat, dimension: 2, start: 0, length: 1, name: nil)
        let zero = graph.multiplication(first, graph.constant(0.0, dataType: .float32), name: nil)
        let padded = graph.concatTensors([flat, zero], dimension: 2, name: nil)
        let sequence = graph.transpose(padded, permutation: [0, 2, 1], name: nil)
        let selected = graph.gather(withUpdatesTensor: sequence, indicesTensor: map,
            axis: 1, batchDimensions: 1, name: nil)
        let channelFirst = graph.transpose(selected, permutation: [0, 2, 1], name: nil)
        return graph.reshape(channelFirst,
            shape: [-1, channels, targetHeight as NSNumber, targetWidth as NSNumber], name: nil)
    }

    func gather(_ graph: MPSGraph, _ source: MPSGraphTensor) -> MPSGraphTensor {
        remap(graph, source, map: gatherMap, sourcePoints: width * width, targetHeight: 1, targetWidth: points.count)
    }

    func restore(_ graph: MPSGraph, _ source: MPSGraphTensor) -> MPSGraphTensor {
        remap(graph, source, map: restoreMap, sourcePoints: points.count, targetHeight: width, targetWidth: width)
    }

    static func attentionBias(
        graph: MPSGraph, geometries: [Int: GraphAttentionGeometry], width: Int,
        tokens: Int, heads: Int, mask: MPSGraphTensor
    ) -> MPSGraphTensor {
        let count = graph.reductionSum(with: mask, axes: [1, 2, 3], name: nil)
        var selected: MPSGraphTensor?
        for size in geometries.keys.filter({ $0 <= width }).sorted(by: >) {
            let geometry = geometries[size]!
            let n = geometry.points.count
            var values = [Float32](repeating: 0, count: heads * tokens * tokens)
            for head in 0..<heads {
                for q in 0..<n {
                    values.replaceSubrange((head * tokens + q) * tokens..<(head * tokens + q) * tokens + n,
                        with: geometry.bias[(head * n + q) * n..<(head * n + q + 1) * n])
                }
            }
            let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
            let tensor = graph.constant(data,
                shape: [1, heads as NSNumber, tokens as NSNumber, tokens as NSNumber], dataType: .float32)
            if let previous = selected {
                let matches = graph.equal(count, graph.constant(Double(n), dataType: .float32), name: nil)
                selected = graph.select(predicate: matches, trueTensor: tensor, falseTensor: previous, name: nil)
            } else {
                selected = tensor
            }
        }
        return selected!
    }
}
