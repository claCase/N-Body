# N-Body
N-Body System Simulation [Wiki Article](https://en.wikipedia.org/wiki/N-body_problem).
<br /><br /> Directed Force Formula between two planets: <br /><br />
![alt-text](https://wikimedia.org/api/rest_v1/media/math/render/svg/f0c5aab28749b00eb610136b76689a0f6cf57976) 
<br /><br /> Position Update Equation: <br /><br />
![alt-text](https://github.com/claCase/N-Body/blob/main/Animations/equation.svg)
<br/><br/> Generate animation parameters: <br/> 
```console
python3 .\main.py --initial "circle" --n_planets 10 --t 2500 --dt 0.02 --trace --trace_len 1700 --save --skip 10
```
![alt-text](https://github.com/claCase/N-Body/blob/main/Animations/animation9.gif)
<br/><br/> Generate animation parameters: <br/> 
```console
python3 .\main.py --initial "spiral" --dt 0.2 --t 1200 --skip 2 --trace --save --skip 3
```
![alt-text](https://github.com/claCase/N-Body/blob/main/Animations/animation8.gif)
```console
python3 .\main.py --initial "uniform" --n_planets 25 --dt 0.2 --t 1200 --skip 3 --save
```
![alt-text](https://github.com/claCase/N-Body/blob/main/Animations/animation10.gif)