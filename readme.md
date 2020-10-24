This is some code to plot flights you've taken on a globe.

![stuff](https://github.com/ricsonc/flight-mapper/blob/main/out.gif?raw=true)

Directly inspired from [gcmap](http://www.gcmap.com/) but with the following features

- higher (arbitrarily high) resolution
- output everything on a spinning globe
- easier customization / colorization
- overlapping trajectories result in a thicker, brighter line
- trajectories have some elevation over the surface
- much slower to run


*What if I want to run the code?*

I recall having lots of problems with installing the dependencies, (perhaps even having to edit some of the cod ein the packages), and also my virtualenv is currently broken so I don't have a list of dependencies at all -- you'll have to figure that out yourself.

Other than that, simply edit data.py such that txts is a list of strings, and each string is a query such as one you might pass in to gcmap -- a very simple example might be `txts = ['SFO-SEA,SFO-YYZ,', 'PHX-OKC,']`. Play around with gcmap to get more familiar with the possibilities. Each string gets plotted in a separate color. For more plotting options, just dig around in the code. Use render to dump all 360 views and generate a video. 
