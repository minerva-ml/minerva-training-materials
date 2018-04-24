graph [
        directed 1
        label "Hello, I am a graph"
        node [
                id 0
                label "First steps"
        ]
        node [
                id 1
                label "Model - localizer"
        ]
        node [
                id 2
                label "Model - classifier"
        ]
        node [
                id 3
                label "Model - aligner"
        ]
        node [
                id 4
                label "Regularization"
        ]
        node [
                id 5
                label "Cross entropy loss"
        ]
        node [
                id 6
                label "Probability calibration transformer"
        ]
        node [
                id 7
                label "Dataset for localization"
        ]
        node [
                id 8
                label "Dataset for keypoint detection"
        ]
        node [
                id 9
                label "Dataset for classification"
        ]
        edge [
                source 0
                target 1
        ]
        edge [
                source 0
                target 5
        ]
        edge [
                source 0
                target 6
        ]
        edge [
                source 0
                target 7
        ]
        edge [
                source 1
                target 2
        ]
        edge [
                source 1
                target 3
        ]
        edge [
                source 1
                target 4
        ]
        edge [
                source 7
                target 8
        ]
        edge [
                source 7
                target 9
        ]
]