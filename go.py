import matplotlib
matplotlib.use('Agg') #need agg for multiple thread render


import cartopy.crs as ccrs
import cartopy
from cartopy import geodesic
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from ipdb import set_trace as st
import shapely.geometry as sgeom
import numpy as np
from munch import Munch as M
from csv import reader
import sys
import time
from PIL import Image

plt.style.use('dark_background')
from pylab import rcParams
rcParams['figure.figsize'] = 20, 20

class PlateCarree(ccrs.PlateCarree):
    @property
    def threshold(self):
        return 0.01

class Orthographic(ccrs.Orthographic):
    @property
    def threshold(self):
        return 1.0

def get_ortho(lat, lon):
    proj = Orthographic(central_latitude=lat, central_longitude=lon)

    # Re-implement the cartopy code to figure out the boundary.
    a = np.float(proj.globe.semimajor_axis or 6378137.0)
    b = np.float(proj.globe.semiminor_axis or a)
    coords = ccrs._ellipse_boundary(a * 0.99999, b * 0.99999, n=7201) #increasing n helps...
    proj._boundary = sgeom.polygon.LinearRing(coords.T)
    return proj

class Airport2Pos(object):
    def __init__(self, fn = '/home/ricson /code/flightmap/airports'):
        self.pos = {}
        with open(fn, 'r') as f:
            cr = reader(f)

            for tokens in cr:
                code = tokens[4].upper()
                lat = float(tokens[6])
                lon = float(tokens[7])
                self.pos[code] = (lon, lat)
                
    def get(self, key):
        if key not in self.pos:
            print('airport code', key, 'not found!')
            # st()
        return self.pos[key]
    
####

def geom2np(line):
    arr = np.array(line[0]) if line else np.zeros((0,2))
    return arr.T

def embiggen(line, scale = 0.0):
    diffs = np.diff(line, axis = 1)
    scales = np.sin(np.linspace(0, np.pi, len(line[0]))) #* len(line[0])
    normals = np.stack([-diffs[1], diffs[0]], axis = 0)
    normals = normals / np.linalg.norm(normals, axis = 0)
    normals = np.concatenate([normals, np.array([0.0, 0.0]).reshape(2,1)], axis = 1)

    #set max_height = 0 for no height plotting..
    max_height = 0.02 + 0.002 * np.sqrt(len(line[0]))
    height_factor = 1 + scales * max_height
    line2 = height_factor * line + normals * scales * scale
    return line2

def heighten(line, scale):
    #return line #ignore everything here...

    diffs = np.diff(line, axis = 1)
    dists = np.linalg.norm(diffs, axis = 0)
    cumdists = np.concatenate((np.zeros(1), np.cumsum(dists)), axis = 0)

    scales = np.sin(cumdists * np.pi / cumdists[-1])

    max_height = 0.025 + 2E-5 * np.sqrt(scale)
    height_factor = 1 + scales * max_height
    return height_factor * line

def kbiggen(line, k, scale):
    lines = []
    for i, x in enumerate(range(-k//2, k//2)):
        if (k % 2) == 0: #even
            x_ = x+0.5
        else:
            x_ = x
        line2 = embiggen(line, scale = x_ * (scale+12000))
        if i % 2: #reverse every other line so we'll have something contiguous
            line2 = line2[:,::-1]
        lines.append(line2)

    return np.concatenate(lines, axis = 1)

def draw_lines(startend, k, info, proj = None):
    sproj, eproj, sprojrev, eprojrev = info
    proj, projrev = proj

    line = sgeom.LineString(startend)

    a, b, c, d = startend[0][0], startend[0][1], startend[1][0], startend[1][1]
    ll = sgeom.LineString(((a,c,), (b,d)))
    length = geodesic.Geodesic().geometry_length(ll) 
    
    linefront = proj.project_geometry(line, ccrs.Geodetic(globe=proj.globe))
    lineback = projrev.project_geometry(line, ccrs.Geodetic(globe=proj.globe))
    
    try:
        linefront_part = geom2np(linefront)
        lineback_part = geom2np(lineback)#[:,::-1]
    except:
        return None

    #flip x of lineback
    lineback_part *= np.array([[-1],[1]])

    #let's do some ad-hoc smoothing, because these don't line up perfectly for some reason
    # if linefront_part.shape[1] > 1 and lineback_part.shape[1] > 1:
    #     #lineback_part = lineback_part[:,1:]
    #     linefront_part[:,-1] = (linefront_part[:,-2] + lineback_part[:,0]) / 2.0
    #     lineback_part[:,0] = (linefront_part[:,-1] + lineback_part[:,1]) / 2.0
    #i think i know how we can get rid of this...
    line_full = np.concatenate([linefront_part, lineback_part], axis = 1)

    flag = True
    while flag:
        for i in range(line_full.shape[1]):
            if i == 0:
                continue

            u, v = line_full[:,i-1], line_full[:,i]
            gap = np.linalg.norm(u-v)
            if gap > 100000.0:
                #print('gap at', i)
                num_pts = int(gap / 10000.0)
                
                #clearly, something is horribly wrong.... we're forced to interpolate here..
                R1 = np.linalg.norm(u)
                R2 = np.linalg.norm(v)

                R = 6378137.0
                if np.abs(R1-R2) > 2E+5:
                    continue
                elif np.abs(R-R1) > 2E+5:
                    continue
                
                TH1 = np.arctan2(*u[::-1])
                TH2 = np.arctan2(*v[::-1])

                rads = np.linspace(R1, R2, num_pts)
                ths = np.linspace(TH1, TH2, num_pts)
                #this might bite us later...
                xs = rads*np.cos(ths)
                ys = rads*np.sin(ths)

                gap_part = np.stack((xs, ys), axis = 0)

                line_full = np.concatenate((line_full[:,:i-1], gap_part, line_full[:,i:]), axis =1)
                break
        else:
            flag = False    

    if line_full.shape[1] == 0:
        return None

    sproj_loc = sproj if sproj is not False else sprojrev
    if not (line_full[:,0] == sproj_loc).all():
        line_full = np.concatenate((np.reshape(sproj_loc, (2,1)), line_full), axis = 1)

    eproj_loc = eproj if eproj is not False else eprojrev
    if not (line_full[:,-1] == eproj_loc).all():
        line_full = np.concatenate((line_full, np.reshape(eproj_loc, (2,1))),axis = 1)

    ###
    
    line_out = heighten(line_full, length)

    #now we do some trimming...
    '''
    each line_full can be split into three segments using the two intersection pts with the surface
    1. if front part, no back part, keep whole line
    2. if back part, no front part, keep only middle segment (if any)
    3. if front and back, then keep first two segments
    '''

    #no trimming
    #return M(line = line_out, count = k)
    
    N = line_out.shape[1]
    if N == 0:
        return None
    
    norms = np.linalg.norm(line_out, axis = 0)
    
    if not lineback: #front only, return everything
        start = None
        end = None
    else: #lineback is true
        
        R = 6378137.0
        if norms.max() < R: 
            first_intersection = None
            second_intersection = None

        else:
            for i in range(N):
                if norms[i] > R:
                    first_intersection = i
                    break
            for i in reversed(range(N)):
                if norms[i] > R:
                    second_intersection = i
                    break

        ###
                
        if linefront: #BOTH, return first two sections
            start = 0
            end = second_intersection
        else: #only back, return middle segment
            if first_intersection is None and second_intersection is None:
                return None #no middle segment
            
            start, end = first_intersection, second_intersection

    return M(line = line_out[:,start:end], count = k)



def parse_and_draw(txts, proj, projrev):

    flyers = {}
    unique = set([])
    pairs = {}
    for flyer, txt in enumerate(txts):
        flights = txt.upper().split(',')
        for flight in flights:
        
            stops = flight.split('-')

            for i, x in enumerate(stops):
                if i == 0:
                    continue
                y = stops[i-1]

                unique.add(x)
                unique.add(y)

                pair = tuple(sorted([x,y]))
                if pair in pairs:
                    pairs[pair] +=1
                else:
                    pairs[pair] = 1

                if pair not in flyers:
                    flyers[pair] = flyer
                    
    getpos = Airport2Pos()
    airport_pos = {x:getpos.get(x) for x in unique}

    argss = []
    for (a,b), v in pairs.items():
        apos, bpos = getpos.get(a), getpos.get(b)

        def in_front(pt, p):
            pos = np.array(p.project_geometry(sgeom.Point(pt)))
            front = not np.isnan(np.array(p.project_geometry(sgeom.Point(pt)))).max()
            return pos if front else front
            
        aproj = in_front(apos, proj)
        bproj = in_front(bpos, proj)
        aprojrev = in_front(apos, projrev)
        bprojrev = in_front(bpos, projrev)
        if aprojrev is not False:
            aprojrev *= np.array([-1,1])
        if bprojrev is not False:
            bprojrev *= np.array([-1,1])

        if aproj is not False:
            argss.append(((apos, bpos), v, (aproj, bproj, aprojrev, bprojrev)))
        else:
            argss.append(((bpos, apos), v, (bproj, aproj, bprojrev, aprojrev)))

        #it's important that if one of these two are in front and the other is in back
        # we put front BEFORE back

    linegroups = [draw_lines(*args, proj = (proj, projrev)) for args in argss]

    
    return linegroups, airport_pos, flyers

def render_and_write(lon, lat, full, flights):

    out_fn = 'out.png'

    lon, lat = float(lon), float(lat)
    proj = get_ortho(lat, lon)
    projrev = get_ortho(-lat, lon + 180.0)

    ax = plt.axes(projection=proj)

    black = [0.3, 0.0, 0.0]
    blue = [0.0, 0.0, 0.2]
    white = [0.3, 0.3, 0.3]

    res = '110m'
    if full:
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', res, edgecolor='red', facecolor='none'), linewidth=0.8)
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'lakes', res, edgecolor='red', facecolor='none'), linewidth=0.8)

    txts = flights.split(';')
    paths, points, flyers = parse_and_draw(txts, proj, projrev)

    for k,(x,y) in points.items():
        alpha = 1.0
        x_, y_ = np.array(proj.project_geometry(sgeom.Point(x,y)))
        if np.isnan(x_) or np.isnan(y_):
            
            x_, y_ = np.array(projrev.project_geometry(sgeom.Point(x,y)))
            x_ = -x_
            assert not np.isnan(x_)
            assert not np.isnan(y_)

            R = 6378137.0
            rr = np.sqrt(x_*x_+y_*y_)
            diff = (R-rr)
            if diff > 5E+4:
                continue
            alpha = 1.0 - diff / 5E+4 #1 -> 0

        plt.plot([x_], [y_], c='white', marker='o', markersize=4, mfc='none', alpha = alpha)
        plt.text(x_, y_, k, c='white', alpha = alpha)
        
    #what is happening here?
    for group, flyer in zip(paths, flyers.values()):
        if group is None:
            continue

        color_dict = ['aqua', 'orange', 'greenyellow', 'hotpink', 'gold', 'plum',
                      'bisque', 'aquamarine', 'palegreen', 'lightcoral', ] + ['white']*50 #in case we go over...
        
        line = group.line
        weight = 1.4+group.count**0.7 #was 0.3...
        plt.plot(*line, linewidth=1.0*weight, c=color_dict[flyer], alpha = 0.25, solid_capstyle='round')

        plt.plot(*line, linewidth=0.3*weight, c=color_dict[flyer], alpha = 1.0, solid_capstyle='round')

    
    #AXES
    
    margin = 0.15
    W = 2*np.pi*1e6*(1 + margin)
    plt.xlim(-W,W)
    plt.ylim(-W,W)

    import matplotlib.ticker as mticker

    tick_res = 10 if full else 45
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=1, color='lime', alpha=1.0, linestyle='dotted')

    gl.ylocator = mticker.FixedLocator(np.linspace(-90., 90., 180//tick_res))
    gl.xlocator = mticker.FixedLocator(np.linspace(0., 360., 360//tick_res))
    
    #CLEAN UP

    ax.outline_patch.set_visible(False)
    ax.background_patch.set_visible(False)
    ax.set_xmargin(0)
    ax.set_ymargin(0)
    ax.axis('off')
    
    dpi = 150 if full else 50
    plt.savefig(out_fn, dpi=dpi, pad_inches=0,bbox_inches='tight')

    if not full:
        img = Image.open(out_fn)
        img = img.resize((img.size[0]*3, img.size[1]*3))
        with open(out_fn, 'wb') as f:
            img.save(f)

    plt.clf()

if __name__ == '__main__':

    t0 = time.time()
    override_long = None
    
    if len(sys.argv) > 1:
        threadid = int(sys.argv[1])
        override_long = float(threadid) - 96.0
        out_fn = 'outs/%d.png' % threadid
        print('starting job', threadid)
    else:
        override_long = -96.0
        out_fn = 'out.png'
    
    #plt.style.use('dark_background')
    FULLRUN = False
    NA = False

    # if not NA:
    #     proj = Orthographic(central_latitude=75.0, central_longitude=-100.0)
    # else:
    #     proj = Orthographic(central_latitude=41.0, central_longitude=-96.0)

    longit = override_long if override_long is not None else 100.0-96.0
    proj = get_ortho(41.0, longit)
    projrev = get_ortho(-41.0, longit + 180.0)

    # from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    # from matplotlib.figure import Figure
    
    ax = plt.axes(projection=proj)

    black = [0.3, 0.0, 0.0]
    blue = [0.0, 0.0, 0.2]
    white = [0.3, 0.3, 0.3]

    res = '50m' if FULLRUN else '110m'
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', res, edgecolor='red', facecolor='none'), linewidth=0.8)
    # ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', res, edgecolor='none', facecolor=blue), linewidth=0.0)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'lakes', res, edgecolor='red', facecolor='none'), linewidth=0.8)

    # ax.add_feature(cartopy.feature.LAND, scale='50m', facecolor='black')
    # ax.add_feature(cartopy.feature.OCEAN, scale='50m', facecolor='grey')

    from data import txts
    #txts = txts[:1]
    
    # print (','.join(txts))

    #DEBUGGING
    #txt = 'nrt-hkg'
    #txt='kul-nrt'

    paths, points, flyers = parse_and_draw(txts, proj, projrev)

    for k,(x,y) in points.items():
        alpha = 1.0
        x_, y_ = np.array(proj.project_geometry(sgeom.Point(x,y)))
        if np.isnan(x_) or np.isnan(y_):
            #continue
            #actually plot both..
            x_, y_ = np.array(projrev.project_geometry(sgeom.Point(x,y)))
            x_ = -x_
            assert not np.isnan(x_)
            assert not np.isnan(y_)

            R = 6378137.0
            rr = np.sqrt(x_*x_+y_*y_)
            diff = (R-rr)
            if diff > 5E+4:
                continue
            alpha = 1.0 - diff / 5E+4 #1 -> 0

            #adjustable alpha... gridlines...

        plt.plot([x_], [y_], c='white', marker='o', markersize=4, mfc='none', alpha = alpha)
        plt.text(x_, y_, k, c='white', alpha = alpha)
        
        # plt.plot([x], [y], c='white', marker='o', markersize=4, mfc='none', transform=ccrs.PlateCarree())
        # plt.text(x, y, k, c='white', transform=ccrs.PlateCarree())
    
    for group, flyer in zip(paths, flyers.values()):
        if group is None:
            continue

        color_dict = ['aqua', 'hotpink', 'plum',
                      'bisque', 'greenyellow', 'aquamarine', 'palegreen', 
                      'lightcoral', 'orange', 'gold']
        
        line = group.line
        weight = 0.3+group.count**0.7 #... controls thickness
        plt.plot(*line, linewidth=1.0*weight, c=color_dict[flyer], alpha = 0.2, solid_capstyle='round')#, marker='o') 

        plt.plot(*line, linewidth=0.3*weight, c=color_dict[flyer], alpha = 1.0, solid_capstyle='round')

    if not NA:
        #margin = 0.03
        margin = 0.15
        
        W = 2*np.pi*1e6*(1 + margin)
        plt.xlim(-W,W)
        plt.ylim(-W,W)
        #uh help
    else:
        plt.xlim(-2.5E6, 2.5E6)
        plt.ylim(-1.8e6, 1.8e6)
        #ax.set_extent([-106, -86, 31, 51])

    #AXES

    import matplotlib.ticker as mticker

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=1, color='lime', alpha=1.0, linestyle='dotted')

    gl.ylocator = mticker.FixedLocator(np.linspace(-90., 90., 18))
    gl.xlocator = mticker.FixedLocator(np.linspace(0., 360., 36))
    #CLEAN UP

    ax.outline_patch.set_visible(False)
    ax.background_patch.set_visible(False)

    ax.set_xmargin(0)
    ax.set_ymargin(0)
    
    ax.axis('off')

    dpi = 300 if FULLRUN else 150

    plt.savefig(out_fn, dpi=dpi, pad_inches=0,bbox_inches='tight')

    print('done in', time.time()-t0)
    #plt.show()
