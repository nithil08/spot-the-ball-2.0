Research in Relation:

Meta SAM3 & SAM3.1: SAM (Segment Anything Model) is a feature released by Meta that is not a research paper but provides a tool to track objects in videos. This feature permits up to 16 objects to be tracked at a time, handling occlusions and motion blur natively.

Spot The Ball: A Benchmark for Visual Social Inference: The foundation of this research line, comparing humans against Vision-Language Models (VLMs) to find an inpainted ball in sports imagery. It proves that humans heavily outperform AI by reading social cues (gaze, body posture), while models rely on weak spatial heuristics like guessing the center of the image.

CoTracker (Meta AI): A transformer-based tracking model that tracks dense points across video sequences jointly rather than independently. By tracking points together over short and long windows, it can predict the location of a completely occluded object based on the trajectory of visible surrounding points.

Scene Informer: A framework tested on the Waymo Open Motion Dataset that uses a transformer to infer occupancy probabilities in hidden zones. It trains AI to read the "social cues" of an environment—such as a visible car slowing down—to infer and predict the trajectory of an occluded object (like a pedestrian or a ball).

Navigation under uncertainty: Autonomous driving research that builds occlusion awareness into behavior prediction using "switching dynamical systems." The AI maintains parallel belief states—one assuming no hidden object exists, and one assuming it does—and dynamically shifts weights between them based on visible contextual clues.

SIV-Bench: A video benchmark evaluating Multimodal Large Language Models (MLLMs) on social scene understanding and dynamics. Its ablation studies prove that when text/audio cues are removed, AI performance plummets, confirming that models struggle severely with purely visual social cues in videos.

SVBench: A benchmark mapping out human social cognition paradigms to evaluate video models. It explicitly tests the AI's grasp on Joint Attention and Social Coordination, showing that while models understand physical motion physics, they fail to track shared human intent or multi-agent strategies.

VRR-QA: A visual relational reasoning benchmark targeting the AI's ability to track "invisible" or implied context across video frames. It demonstrates that cutting-edge video models fail at tasks requiring implicit spatial reasoning across time, showing a massive deficiency in temporal object permanence when driven by context alone.

VideoGaze (MIT CSAIL): A specialized computer vision framework that tracks and maps human gaze vectors across unconstrained videos. It estimates 3D head orientation to create a spatial probability density map, allowing an architecture to mathematically triangulate where an occluded object is located based purely on where people are looking.
