# Simulating 2-D Wildland Fire Behavior 
## by Diane Wang

![Lattice QCD](https://besthqwallpapers.com/Uploads/5-10-2017/22805/thumb2-dragon-warriors-fire-battle-art.jpg)

---
# Abstract

This project simulated the 2-D wildland wind-driven fire behavior in different conditions. Both analytical ways and cellular automata are implemented in this project. The fire on the homogeneous field with no wind spread in a circle. And if there's a constant wind (velocity and direction), the fire spread in an elliptical shape. Then, if the constant wind has a direction change, the fire spread direction changed with the wind. When simulating fire with the approach of cellular automata, the fires are treated as cells, and, therefore, it is important to know how the neighbors of each cell get burned or not. For the fire CA function, the direction of the wind is needed. The direction is chosen from 'N', 'S', 'E', 'W', 'NW', 'NE', 'SW', 'SE'. The fire spread in the direction of the wind, which demonstrated the dominated effects of wind when other effects are neglected.

----
# Statement of Need

For the safety of firefighters and the preservation of forest resources, it is imperative to predict wildland fire behavior accurately. However, many operational models neglect the possibility of substantial feedbacks between the ﬁre and the atmosphere above it. We are planning to develop a 2-D wildland wind-driven fire behavior model coupled with the atmospheric model, Advanced Regional Prediction System (ARPS). This project focused on how to simulate fire behavior based on the wind in different conditions. Future work is to output the heat flux from the fire model into the ARPS.

----
# Installation instructions
Instructions are shown in makefile.
- "To get started create an environment using:"
    - "	make init"
- "	conda activate ./envs"

- "To generate project documentation use:"
    - "	make doc"

- "To Lint the project use:"
    - "	make lint"

- "To run unit tests use:"
    - "	make test"

# Unit Tests
- "To run unit tests use:"
    - " make test"
    
The tests folder is inside /fire_simulation_cmse802/fire_behavior. Run "pytest" in this folder works, too!



### check fire_behavior/Example.ipynb for details about how to get this software work.

### Here's a video for more information about this project: https://youtu.be/ocxiM1hJD7A

---
# Methodology

Generally, two main approaches to representing fire behaviors have been implemented in several simulation models. The first one is considering the fire as a set of continuous but independent cells that spreads in the growth of number, which is described as a raster implementation. The second one holds the view that the fire perimeter is a closed curve of linked points, which is described as a vector implementation. Also, two main approaches to representing the propagation of the fire with some forms of expansion algorithm are used in these models. The first one expands the perimeter based on a direct-contact or near-neighbour proximity. The second one is based on Huygens’wavelet principle in which each point on the perimeter becomes the source of the interval of fire spread. This project achieved 2-D wildland wind-driven fire behavior simulation in both analytical ways and cellular automata. 

---
# Concluding Remarks

This project achieved 2-D wildland wind-driven fire behavior simulation. Future work is to output the heat flux from the fire model to get coupled with the atmopheric model, ARPS.

----
# References

#### PROJECT LINK: https://gitlab.msu.edu/wangti68/fire_simulation_cmse802.git
- Anderson, Hal E. 1983. Predicting wind-driven wild land fire size and shape. Research Paper INT-RP-305. Ogden, UT: USDA Forest Service, Intermountain Forest and Range Experiment Station. 26 p.
- Alexandridis, A., D. Vakalis, C.I. Siettos, and G.V. Bafas. “A Cellular Automata Model for Forest Fire Spread Prediction: The Case of the Wildfire That Swept through Spetses Island in 1990.” Applied Mathematics and Computation 204, no. 1 (October 2008): 191–201.
- Dahl, N., Xue, H., Hu, X. et al. (2015) Coupled ﬁre–atmosphere modeling of wildland ﬁre spread using DEVS-FIRE and ARPS. Nat Hazards 77:1013–1035. https://doi.org/10.1007/s11069-015-1640-y
- Rothermel RC (1972) A mathematical model for predicting fire spread in wildland fuels. USDA Forest Service, Intermountain Forest and Range Experimental Station, Research Paper INT-115. (Odgen, UT)