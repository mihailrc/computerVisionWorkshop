Slides:
- Show of hands:
	- How many of you can write code
	- How many of you are familiar with the term: AI/ML/Deep Learning
	- How many of you think using video to count cars or identify potholes is an easy problem?
- Goals
	- Lean how to solve Computer Vision problems
	- Use provided toolkit to experiment and ...
	- Apply what you learned to solve your own problems
- Traditional Programming vs ML
	- Manually write the logic. Input + Rules => Outputs
	- What if problem is so hard you cannot write the rules?
		- Learn from data
		- Inputs + Outputs => Model (or Rules)
- AI/ML/Deep Learning
	- AI - create intelligent machines that can simulate thinking capability and behavior. E.g robotics
	- ML - subset of AI that allow machines to learn from data without being programmed explicitly. Eg. predicting product preferences.
	- Deep Learning - subset of ML that uses complex artificial neural networks. ML features manually created, in DL features are automatically extracted. E.g. image classification or generating images from a description
    - https://towardsdatascience.com/notes-on-artificial-intelligence-ai-machine-learning-ml-and-deep-learning-dl-for-56e51a2071c2
    - https://medium.com/@prolinecoders/diffrerence-between-artificial-intelligence-machine-learning-and-deep-learning-533a6373dabe
- Quiz: Deep Learning, ML, AI or Traditional Programming?
	- Lane Detection https://youtu.be/OV1l5xFHiwQ?t=16
	- Path Planning https://youtu.be/t7P4au6bT1Q?t=23
	- Vehicle Detection https://youtu.be/kZViHcQgGYU?t=37
	- Behavioral Cloning Video https://youtu.be/vXzgXb1rkjw?t=24
- Biggest Challenge
	- Images/video represent unstructured data that is hard to analyze and interpret automatically.
	- What  can be done about it? How do we turn unstructured information into structured?
- How Computers learn to recognize objects
	- https://youtu.be/Cgxsv1riJhI
- Our end product
	- https://youtu.be/SdSO5bvIDAQ
- YOLO
	- basic intro to YOLO
	- how it works
- Coding: Building block 1 -  Detection algorithm
	- Coding: input image -> outputs: identified objects, bounding boxes, confidence scores
- Coding: Building block 2 -  Open CV
	- write text, boxes
	- use image from previous step. Draw boxes, analyze results
- Coding Inference on video - understanding performance
  - Performance:
    - Trained Models
    - CPU vs GPU
  - Configuration parameters
    - classes
    - confidence threshold
- Coding: Object tracking
  - frames are independent. How do you relate objects across frames?
  - Challenges
  - Different types of models
  - Kalman Filters
  - Tuning parameters
- Coding: Vehicle Counts
	- Think about how to implement the solution
	- code walkthrough
- Coding: Vehicle Counts per lane
- Coding: Vehicle Counts per vehicle type - maybe skip?
- Discussion or coding:
	- Calculate vehicle speed
	- Traffic flow analytics
- Custom model - use existing data
	- Start with existing model if you can find one
	- Train the model
	- Evaluate the model
- Custom model - annotate your own data
	- Image annotations
	- Data format
- Active Learning
	- improve model overtime
	- adapt to changes
- Some recent advances
  - Stable Diffusion
    - https://twitter.com/xsteenbrugge/status/1558508866463219712
    - https://github.com/CompVis/stable-diffusion
  - Whisper
    - https://openai.com/blog/whisper/
- Wrap-up
- Other
 - DL vs Humans https://twitter.com/fchollet/status/1573843771548389376
 - Nvidia nano: https://nvidianews.nvidia.com/news/nvidia-jetson-orin-nano-sets-new-standard-for-entry-level-edge-ai-and-robotics-with-80x-performance-leap
  - https://developer.nvidia.com/blog/solving-entry-level-edge-ai-challenges-with-nvidia-jetson-orin-nano/
