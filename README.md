# ROAR Simulation Racing Submission

This repository contains a working solution for the **Robot Open Autonomous Racing (ROAR) Simulation Racing Series**.  The competition is run using the official `ROAR_Competition` starter code, and your solution must be implemented in `submission.py` under the `competition_code` folder.  This repository follows that structure and includes the unmodified skeleton files from the official starter kit along with a custom solution.

## Structure

```
ROAR_SimRace_Submission/
├── competition_code/
│   ├── competition_runner.py    # Provided skeleton – do not modify
│   ├── infrastructure.py        # Provided skeleton – do not modify
│   └── submission.py            # Your autonomous agent implementation
├── instruction.txt              # Any special instructions for the judges
└── README.md                    # This file
```

### `competition_runner.py` and `infrastructure.py`

These files come directly from the official ROAR competition skeleton.  They set up the environment, spawn the vehicle and sensors, and wrap the agent for evaluation.  **Do not modify these files**; they are included here unchanged so that the organisers can run your agent without any surprises.

### `submission.py`

The heart of your solution lives in `submission.py`.  Inside the `RoarCompetitionSolution` class you’ll find two asynchronous methods:

* `initialize()` – called once before the race starts.  Here you can perform any initial computation such as selecting a starting waypoint.
* `step()` – called every simulation tick.  In this method you read the latest observations and decide on a control action to apply to the vehicle.

Our implementation uses a simple proportional controller that steers towards a look‑ahead waypoint and adjusts throttle based on the vehicle’s speed and how sharply it needs to turn.  The look‑ahead distance and target speed have been tuned relative to the original starter code to provide smoother driving on the Monza track.

## Instructions

To run this solution you will need to follow the official ROAR competition installation instructions:

1. Clone the `ROAR_PY` repository and install all dependencies.  The guide can be found on the competition website【569463672377830†L95-L113】.
2. Clone or fork this repository into your own GitHub account and ensure the `competition_code` folder is present.  You must keep the skeleton files intact.
3. (Optional) Create a conda environment with Python 3.8, install `carla==0.9.12` and other packages as described in the competition documentation【525127204158809†screenshot】.
4. Start the Monza map executable and run your agent using `competition_runner.py` from inside the `ROAR_PY_Competition` environment.

## Notes

* This solution targets the Monza map V1.1 used in the Summer 2025 competition【569463672377830†L43-L46】.
* The code here cannot be executed on machines without a supported GPU; however, you can read and modify the logic from any platform.  The organisers will run your agent on their own hardware during evaluation.
* The `instruction.txt` file is intentionally brief because this solution does not require any additional Python packages beyond those already included in the official starter kit.
