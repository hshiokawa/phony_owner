import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import walker_utils as wu
from itertools import combinations
plt.rcParams['interactive'] = True
from gatspy.periodic import LombScargleMultibandFast
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


g_n = 9.80665 # earth gravity

def get_data(ID):
    return( pd.read_csv( 'data/%02d.csv'%ID, names = [ 't', 'ax', 'ay', 'az' ]) )

def get_data_file(fname):
    return( pd.read_csv( fname, names = [ 't', 'ax', 'ay', 'az' ]) )


#=============================
# parsing data into DT, Calculate new variables
class parse_data(object):

    def __init__( self, data, DT ):
        # Segment size
        self.DT = DT
        # Parse raw data
        self.data_parsed = self.parsing( data )
        # Calculate average and standard deviation of each timesegment.
        # The most basic features.
        self.t_parsed_avg,\
        self.data_parsed_avg,\
        self.sig_parsed_avg\
            = self.averaging( self.data_parsed )
        # Transform acceleration in direction of earth's coordinate (z: direc. of gravity)
        self.t_parsed,\
        self.a_parsed,\
        self.ax_parsed,\
        self.ay_parsed,\
        self.az_parsed\
            = self.get_a( self.data_parsed )
        # Play around with available variables to create various features
        self.feature_enginearing()


    # Engineer various features, getting messier as I try more and more....
    def feature_enginearing( self ):
        # Prepare empty lists for appending
        self.ax_parsed_avg = []; self.ay_parsed_avg = []; self.az_parsed_avg = []
        self.sigax_parsed_avg = []; self.sigay_parsed_avg = []; self.sigaz_parsed_avg = []
        self.siga_hor_parsed_avg = []; self.siga_ver_parsed_avg = []
        # Go over all the segments, calculate features of each timesegment and append
        for iseg in range( len(self.t_parsed) ):
            hor = np.sqrt( self.ax_parsed[iseg]**2. + self.ay_parsed[iseg]**2. )
            self.siga_hor_parsed_avg.append( np.std(hor) )
            self.siga_ver_parsed_avg.append( np.std(self.az_parsed[iseg]) )
            self.ax_parsed_avg.append( np.average(self.ax_parsed[iseg]))
            self.ay_parsed_avg.append( np.average(self.ay_parsed[iseg]))
            self.az_parsed_avg.append( np.average(self.az_parsed[iseg]))
            self.sigax_parsed_avg.append( np.std(self.ax_parsed[iseg]))
            self.sigay_parsed_avg.append( np.std(self.ay_parsed[iseg]))
            self.sigaz_parsed_avg.append( np.std(self.az_parsed[iseg]))
        self.siga_hor_parsed_avg = np.array( self.siga_hor_parsed_avg )
        self.siga_ver_parsed_avg = np.array( self.siga_ver_parsed_avg )
        self.ax_parsed_avg = np.array( self.ax_parsed_avg )
        self.ay_parsed_avg = np.array( self.ay_parsed_avg )
        self.az_parsed_avg = np.array( self.az_parsed_avg )
        self.sigax_parsed_avg = np.array( self.sigax_parsed_avg )
        self.sigay_parsed_avg = np.array( self.sigay_parsed_avg )
        self.sigaz_parsed_avg = np.array( self.sigaz_parsed_avg )

        self.a_parsed_avg,\
        self.siga_parsed_avg,\
        self.period1_parsed_avg,\
        self.period2_parsed_avg,\
        self.power1_parsed_avg,\
        self.power2_parsed_avg,\
        self.freq_parsed,\
        self.power_parsed,\
            = self.get_peaks( self.t_parsed, self.a_parsed )

        self.freq_top3 = []
        self.power_top3 = []
        for f, p in zip( self.freq_parsed, self.power_parsed ):
            f_top3, p_top3 = wu.get_top3( f, p )
            self.freq_top3.append( f_top3 )
            self.power_top3.append( p_top3 )
        self.freq_top3 = np.array( self.freq_top3 )
        self.power_top3 = np.array( self.power_top3 )

        self.newvar1 = np.transpose(self.freq_top3)[0]
        self.newvar2 = np.transpose(self.freq_top3)[1]
        self.newvar3 = np.transpose(self.freq_top3)[2]
        self.newvar4 = np.transpose(self.power_top3)[0]
        self.newvar5 = np.transpose(self.power_top3)[1]
        self.newvar6 = np.transpose(self.power_top3)[2]


    def parsing( self, data ):
        t_s = data['t'][0]
        t_e = t_s + self.DT
        i_s = 0
        data_parsed = []
        # need to loop since it's unevely sampled...
        for i, t in enumerate(data['t']):
            if t > t_e:
                data_parsed.append(data.iloc[i_s:i,:])
                i_s = i
                t_e = t + self.DT
        return data_parsed

    def averaging( self, data_parsed ):
        # calculate average and standard dev. for each segment
        t_avg = []; d_avg = []; sig = []
        for d in data_parsed:
            t_avg.append( np.average(d['t']) )
            cols = d.iloc[:,1:]
            d_avg.append( [np.average(cols[col]) for col in cols] )
            sig.append(   [np.std(    cols[col]) for col in cols] )
        d_avg = list( map( list, zip(*d_avg) ) )
        sig   = list( map( list, zip(*sig  ) ) )
        return np.array(t_avg),np.array(d_avg),np.array(sig)


    # The 'a' parameter, sqrt( sum_i( (a_i-avg(a_i))^2 ) ) for all the segments.
    def get_a( self, data_parsed ):
        n_seg = len(data_parsed)
        a=[]; ax=[]; ay=[]; az=[]; t=[]
        for i_seg in range( n_seg ):
            a_x = data_parsed[i_seg]['ax']
            a_y = data_parsed[i_seg]['ay']
            a_z = data_parsed[i_seg]['az']
            abar_x = np.average( a_x )
            abar_y = np.average( a_y )
            abar_z = np.average( a_z )

            # Set convenient constants for coordinate transformation
            gax_tmp = g_n**2 - abar_x**2
            if gax_tmp <=0.:
                gax_tmp = g_n
            else:
                gax = np.sqrt(gax_tmp)

            # Cordinate transformation:
            # Cellphone x,y,z -> Earth direction of gravity (z) and horizontal direction
            ae_x = ( a_x*gax**2 - a_y*abar_x*abar_y - a_z*abar_x*abar_z )  /  ( g_n * gax )
            ae_y = ( a_y*abar_z - a_z*abar_y ) /  gax
            ae_z = ( a_x*abar_x + a_y*abar_y + a_z*abar_z ) / g_n

            a.append( np.sqrt(ae_x**2 + ae_y**2 + ae_z**2) )
            ax.append( ae_x )
            ay.append( ae_y )
            az.append( ae_z )
            t.append( data_parsed[i_seg]['t'] )
        return( np.array(t), np.array(a), np.array(ax), np.array(ay), np.array(az) )


    def get_peaks( self, t, a ):
        a_seg=[]; stda_seg=[];
        period1_seg=[]; period2_seg=[]
        ffseg=[]; fpseg=[];
        power1_seg=[]; power2_seg=[]

        def get_fp( ti, ai ):
            f, p = wu.get_LS([ ti, ai ]) # freq. vs. power
            li = (f>1/2.)*(f<1/0.1) # period faster than 2sec, slower than 0.1sec
            ff=f[li]; fp=p[li] # filtered freq. vs. power
            fp_isort = np.argsort(fp)
            period1 = ff[fp_isort[-1]] # freq. of the strongest power
            period2 = ff[fp_isort[-2]] # freq. of the 2nd strongest power
            power1 = fp[fp_isort[-1]]
            power2 = fp[fp_isort[-2]]
            return ff, fp, period1, period2, power1, power2

        for iseg in range(len(t)):
            ti = np.array(t[iseg]); ai = np.array(a[iseg])
            ff, fp, period1, period2, power1, power2 = get_fp( ti, ai ) # freq. vs. power, period
            a_seg.append(np.average(ai))
            stda_seg.append(np.std(ai))
            period1_seg.append(period1)
            period2_seg.append(period2)
            power1_seg.append(power1)
            power2_seg.append(power2)
            ffseg.append(ff); fpseg.append(fp); ###
        return np.array(a_seg),np.array(stda_seg),\
            np.array(period1_seg),np.array(period2_seg),\
            np.array(power1_seg),np.array(power2_seg),\
            np.array(ffseg),np.array(fpseg)

    def cleaning( self ):
        # Remove non-steady part of the parsed data
        n_seg = len( self.data_parsed ) # total number of segments
        idx = [True]*n_seg
         # remove first 2 and last 2 segments
        idx[0] = False; idx[1] = False; idx[n_seg-1] = False; idx[n_seg-2] = False

        # go over all the segments, compare with 1 index back
        variation_lim = 2.
        for i in range( 1, n_seg ):
            for j in range(3): # x,y,z direc
                if abs(self.data_parsed_avg[j][i-1] - self.data_parsed_avg[j][i]) > variation_lim: idx[i] = False
                if abs(self.sig_parsed_avg[j][i-1] - self.sig_parsed_avg[j][i]) > variation_lim: idx[i] = False
        # go over all the segments, compare with 1 index front
        for i in range( 0, n_seg-1 ):
            for j in range(3): # x,y,z direc
                if abs(self.data_parsed_avg[j][i+1] - self.data_parsed_avg[j][i]) > variation_lim: idx[i] = False
                if abs(self.sig_parsed_avg[j][i+1] - self.sig_parsed_avg[j][i]) > variation_lim: idx[i] = False
        # go over all the segments, and remove points based on its properties
        for i in range( 2, n_seg-2 ):
            if self.power1_parsed_avg[i] < 12.: idx[i] = False
            if any(np.array([idx[i-2],idx[i],idx[i+1],idx[i+2]])==True)==False: idx[i] = False

        idx = np.array(idx)
        self.t_parsed_avg    = np.array(self.t_parsed_avg[idx])
        self.data_parsed_avg = np.array([self.data_parsed_avg[0][idx], self.data_parsed_avg[1][idx], self.data_parsed_avg[2][idx]])
        self.sig_parsed_avg  = np.array([self.sig_parsed_avg[1][idx], self.sig_parsed_avg[1][idx], self.sig_parsed_avg[2][idx]])

        self.a_parsed_avg = np.array(self.a_parsed_avg[idx])
        self.ax_parsed_avg = np.array(self.ax_parsed_avg[idx])
        self.ay_parsed_avg = np.array(self.ay_parsed_avg[idx])
        self.az_parsed_avg = np.array(self.az_parsed_avg[idx])
        self.sigax_parsed_avg = np.array(self.sigax_parsed_avg[idx])
        self.sigay_parsed_avg = np.array(self.sigay_parsed_avg[idx])
        self.sigaz_parsed_avg = np.array(self.sigaz_parsed_avg[idx])
        self.siga_parsed_avg = np.array(self.siga_parsed_avg[idx])
        self.siga_hor_parsed_avg = np.array(self.siga_hor_parsed_avg[idx])
        self.siga_ver_parsed_avg = np.array(self.siga_ver_parsed_avg[idx])
        self.period1_parsed_avg = np.array(self.period1_parsed_avg[idx])
        self.period2_parsed_avg = np.array(self.period2_parsed_avg[idx])
        self.power1_parsed_avg = np.array(self.power1_parsed_avg[idx])
        self.power2_parsed_avg = np.array(self.power2_parsed_avg[idx])

        self.freq_parsed = np.array(self.freq_parsed[idx])
        self.power_parsed = np.array(self.power_parsed[idx])
        self.freq_top3 = np.array(self.freq_top3[idx])
        self.power_top3 = np.array(self.power_top3[idx])
        self.newvar1 = np.array(self.newvar1[idx])
        self.newvar2 = np.array(self.newvar2[idx])
        self.newvar3 = np.array(self.newvar3[idx])
        self.newvar4 = np.array(self.newvar4[idx])
        self.newvar5 = np.array(self.newvar5[idx])
        self.newvar6 = np.array(self.newvar6[idx])


#=============================

def accuracy_check():

    DT = 5. # segment size in second
    ID = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,21]
    features,labels = get_features_and_labels(DT,ID)

    avg = 0.
    navg = 100
    for cnt in range(navg):
        features_train, features_test, labels_train, labels_test =\
            train_test_split(features, labels, test_size=0.3, random_state=int(random.random()*1000))

        # Classifiers
        #clf = KNeighborsClassifier(n_neighbors=5)
        #clf = DecisionTreeClassifier(min_samples_split=2)
        clf = SVC(C=10)#,kernel='poly')
        #clf = RandomForestClassifier(min_samples_split=3)

        clf.fit( features_train, labels_train )
        labels_pred = clf.predict( features_test )
        acc = accuracy_score(labels_pred,labels_test)
        f1s = f1_score(labels_pred,labels_test,average='micro')
        avg += f1s
    avg /= navg
    print('Average F1 Score = %f'%avg)


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def plot_parse( par, f ):
    t = par.t_parsed_avg; data = par.data_parsed_avg; sig = par.sig_parsed_avg
    f.errorbar( t, data[0], yerr=sig[0], fmt='.', c='b', ms=10, label='ax' )
    f.errorbar( t, data[1], yerr=sig[1], fmt='.', c='g', ms=10, label='ay' )
    f.errorbar( t, data[2], yerr=sig[2], fmt='.', c='r', ms=10, label='az' )
    f.plot( t, np.sqrt(data[0]**2+data[1]**2+data[2]**2), 'ko-', lw=2., label='Ma' )
    plt.legend( loc='upper right' )

def plot_data( data, f ):
    f.plot( data['t'], data['ax'], alpha=0.3, c='b', label="ax") # -1 for 0 index
    f.plot( data['t'], data['ay'], alpha=0.3, c='g', label="ay")
    f.plot( data['t'], data['az'], alpha=0.3, c='r', label="az")
    f.set_xlabel( 'Time [sec]' ); f.set_ylabel( r'Acceleration [m/$s^2$]' )

def plot_ai( par, f ):
    for iseg in range( len(par.t_parsed) ):
        f.plot( par.t_parsed[iseg], par.ax_parsed[iseg], alpha=0.2, c='b')
        f.plot( par.t_parsed[iseg], par.ay_parsed[iseg], alpha=0.3, c='r')
        f.plot( par.t_parsed[iseg], par.az_parsed[iseg], alpha=0.3, c='g')
    f.plot( par.t_parsed_avg, par.ax_parsed_avg, 'bo-', label="ax")
    f.plot( par.t_parsed_avg, par.ay_parsed_avg, 'ro-', label="ay")
    f.plot( par.t_parsed_avg, par.az_parsed_avg, 'go-', label="az")
    f.set_xlabel( 'Time [sec]' ); f.set_ylabel( r'Acceleration [m/$s^2$]' )


varlist = [ 'siga_hor_parsed_avg', 'siga_ver_parsed_avg', 'sigaz_parsed_avg', 'newvar1', 'newvar2', 'newvar3', 'newvar4', 'newvar5', 'newvar6' ]
#vars_id_use = [0,1]

def store_data( name, DT ):
    global hotaka
    if name=='hotaka':
        direc = 'data_B/'
        #fnames = [ direc+'hotaka_panda-alewife_01-17.csv',\
        #           direc+'hotaka_stopandshop-home_01-16.csv',\
        #           direc+'hotaka_southstation-insight_01-16.csv',\
        #           direc+'hotaka_walk.csv',\
        #           direc+'hotaka_southstation-insight_01-17.csv' ]
        #fnames = [ direc+'Hotaka_01.csv',\
        #           direc+'Hotaka_02.csv',\
        #           direc+'Hotaka_03.csv' ]
        #fnames = [ direc+'Hotaka_03.csv' ]
        fnames = [ direc+'Hotaka_B1.csv',\
                   direc+'Hotaka_B2.csv',\
                   direc+'Hotaka_B3.csv',\
                   direc+'Hotaka_B4.csv',\
                   direc+'Hotaka_B5.csv',\
                   direc+'Hotaka_B6.csv' ]

        var = [[] for i in range(len(varlist))]
        for i, fn in enumerate( fnames ):
            print( "reading %s ..."%fn )
            data = get_data_file( fn )
            par = parse_data( data, DT )
            par.cleaning()

            for j, v in enumerate(varlist):
                #var_tmp[j] += list(par.v)
                exec( "var["+str(j)+"]+=" + "list(par."+v+")" )
        hotaka = np.array(var)
        f = open(direc+"hotaka_%dD.dat"%len(varlist),"w")
        for i in range(len(hotaka[0])):
            for j in range(len(hotaka)):
                f.write("%le "%hotaka[j][i])
            f.write("\n")
        f.close()


def store_hotaka():
    global hotaka
    hotaka = np.transpose(np.loadtxt("../phony_owner_data/data/hotaka_9D.dat"))


def loop_vars(name):
    id_list = [0,1,2,5,6,8]
    all_list = []
    for i in range(2,len(id_list)):
        all_list += list(combinations(id_list,i))

    acc_all = []
    for vars_id_use in all_list:
        acc = walker_identifier_acc('hotaka','../phony_owner_data/data/'+name+'.csv',vars_id_use)
        acc_all.append( acc )
        print(name,vars_id_use, acc)
    res = sorted(zip(acc_all,all_list))
    f = open(name+"_all.dat","w")
    for r in res:
        f.write("%f "%(r[0]))
        for i in range(len(r[1])):
            f.write("%d "%r[1][i])
        f.write("\n")


def loop_vars_all():
    loop_vars('Liz')
    loop_vars('Andrew')
    loop_vars('DaveH')
    loop_vars('Godine')
    loop_vars('Jason')
    loop_vars('Jasmine')
    loop_vars('Ryan')
    loop_vars('Tanya')
    loop_vars('Vlad')
    loop_vars('Wei')
    loop_vars('Hotaka_B1')
    loop_vars('Hotaka_B2')
    loop_vars('Hotaka_B3')
    loop_vars('Hotaka_B4')
    loop_vars('Hotaka_B5')
    loop_vars('Hotaka_B6')


def walker_identifier_acc( name1, name2, vars_id_use ):

    if name1=='hotaka':
        try: vars1 = hotaka
        except Exception:
            print("Did you call store_data( name, DT ) ?")
            return
    else:
        print("Not yet...")
        return

    Np = len(vars1[0]) # total number of data points
    # choose actual variables to use
    N_feas = len( vars_id_use ) # total number of features
    feas1 = []
    for i in vars_id_use: feas1.append( vars1[i] )
    lab = np.ones( Np ) # labels

    # Feature rescaling
    feas1_avg=[]; feas1_ext=[];
    for i in range( N_feas ):
        feas1_avg.append( np.average( feas1[i] ) )
        feas1_ext.append( np.std( feas1[i] ) )
        feas1[i] = (feas1[i] - feas1_avg[i]) / feas1_ext[i]

    # Data to be examined
    DT = 3.
    data = get_data_file( name2 )
    par = parse_data( data, DT )
    par.cleaning()

    vars2 = []
    for v in varlist: exec( "vars2.append( par."+v+" )" )
    # Feature rescaling
    #for i in range(len(vars2)):
    #    vars2[i] = vars2[i] - np.average(vars2[i])/np.std(vars2[i])
    # choose actual variables to use
    feas2 = []
    for i in vars_id_use: feas2.append( vars2[i] )

    # Feature rescaling
    for i in range( N_feas ):
        feas2[i] = (feas2[i] - feas1_avg[i]) / feas1_ext[i]


    # Make background "No" points and label them
    #fac = .5 # adjustment factor of backgound points number density
    #h = 1./Np**(1./N_feas) # choose step size so that total # of "No" points ~ Np
    dim_minmax = []
    for i in range( N_feas ):
        dim_minmax.append([min(min(feas1[i]),min(feas2[i])), max(max(feas1[i]),max(feas2[i]))])

    # Original command for 2 features:
    # xx,yy = np.meshgrid(np.arange(xmin_glob,xmax_glob,hx/fac),np.arange(ymin_glob,ymax_glob,hy/fac))
    dim_axes = [[] for i in range( N_feas )]
    h = []
    fac = 1.
    cmd = ""
    for i in range( N_feas ): cmd += "dim_axes[%d],"%i
    cmd += " = np.meshgrid( "
    for i in range( N_feas ):
        xmin = dim_minmax[i][0]; xmax = dim_minmax[i][1]
        h.append( (xmax - xmin)/Np**(1./N_feas) )
        #cmd += "np.arange(dim_minmax[%d][0],dim_minmax[%d][1],),"%(i,i)
        cmd += "np.arange(%le,%le,%le),"%(xmin,xmax,h[i]/fac)
    cmd = cmd[:-1]+" )"
    exec(cmd)

    # Original command for 2 features: fea_bg = np.c_[xx.ravel(),yy.ravel()]
    cmd = "feas_bg = np.c_["
    for i in range( N_feas ): cmd += "dim_axes[%d].ravel(),"%i
    cmd = cmd[:-1]+"]"
    exec(cmd)
    lab_bg = np.zeros(len(feas_bg))

    # Combine real data points and background points
    feas1 = np.transpose(feas1)
    features = np.array(list(feas1) + list(feas_bg))
    labels = np.array(list(lab) + list(lab_bg))

    # Train model
    clf = KNeighborsClassifier(n_neighbors=5)
    #clf = SVC(C=100)
    #clf = RandomForestClassifier(min_samples_split=3)
    #clf = DecisionTreeClassifier(min_samples_split=C)
    clf.fit(features,labels)

    # Examine new data
    cmd = "res = clf.predict( np.c_["
    for i in range( N_feas ): cmd += "feas2[%d],"%i
    cmd = cmd[:-1]+" ] )"
    exec(cmd)

    acc = 100*sum(res)/len(res)
    return acc


def walker_identifier( name1, name2, vars_id_use ):

    if name1=='hotaka':
        try: vars1 = hotaka
        except Exception:
            print("Did you call store_data( name, DT ) ?")
            return
    else:
        print("Not yet...")
        return


    Np = len(vars1[0]) # total number of data points
    # choose actual variables to use
    N_feas = len( vars_id_use ) # total number of features
    feas1 = []
    for i in vars_id_use: feas1.append( vars1[i] )
    lab = np.ones( Np ) # labels

    # Feature rescaling
    feas1_avg=[]; feas1_ext=[];
    for i in range( N_feas ):
        feas1_avg.append( np.average( feas1[i] ) )
        feas1_ext.append( np.std( feas1[i] ) )
        feas1[i] = (feas1[i] - feas1_avg[i]) / feas1_ext[i]

    # Data to be examined
    DT = 3.
    print( "reading %s ..."%name2 )
    data = get_data_file( name2 )
    par = parse_data( data, DT )
    par.cleaning()

    vars2 = []
    for v in varlist: exec( "vars2.append( par."+v+" )" )
    # Feature rescaling
    #for i in range(len(vars2)):
    #    vars2[i] = vars2[i] - np.average(vars2[i])/np.std(vars2[i])
    # choose actual variables to use
    feas2 = []
    for i in vars_id_use: feas2.append( vars2[i] )

    # Feature rescaling
    for i in range( N_feas ):
        feas2[i] = (feas2[i] - feas1_avg[i]) / feas1_ext[i]


    # Make background "No" points and label them
    #fac = .5 # adjustment factor of backgound points number density
    #h = 1./Np**(1./N_feas) # choose step size so that total # of "No" points ~ Np
    dim_minmax = []
    for i in range( N_feas ):
        dim_minmax.append([min(min(feas1[i]),min(feas2[i])), max(max(feas1[i]),max(feas2[i]))])

    # Original command for 2 features:
    # xx,yy = np.meshgrid(np.arange(xmin_glob,xmax_glob,hx/fac),np.arange(ymin_glob,ymax_glob,hy/fac))
    dim_axes = [[] for i in range( N_feas )]
    h = []
    fac = 1.
    cmd = ""
    for i in range( N_feas ): cmd += "dim_axes[%d],"%i
    cmd += " = np.meshgrid( "
    for i in range( N_feas ):
        xmin = dim_minmax[i][0]; xmax = dim_minmax[i][1]
        h.append( (xmax - xmin)/Np**(1./N_feas) )
        #cmd += "np.arange(dim_minmax[%d][0],dim_minmax[%d][1],),"%(i,i)
        cmd += "np.arange(%le,%le,%le),"%(xmin,xmax,h[i]/fac)
    cmd = cmd[:-1]+" )"
    exec(cmd)

    # Original command for 2 features: fea_bg = np.c_[xx.ravel(),yy.ravel()]
    cmd = "feas_bg = np.c_["
    for i in range( N_feas ): cmd += "dim_axes[%d].ravel(),"%i
    cmd = cmd[:-1]+"]"
    exec(cmd)
    lab_bg = np.zeros(len(feas_bg))

    # Combine real data points and background points
    feas1 = np.transpose(feas1)
    features = np.array(list(feas1) + list(feas_bg))
    labels = np.array(list(lab) + list(lab_bg))

    # Train model
    clf = KNeighborsClassifier(n_neighbors=5)
    #clf = SVC(C=100)
    #clf = RandomForestClassifier(min_samples_split=3)
    #clf = DecisionTreeClassifier(min_samples_split=C)
    clf.fit(features,labels)

    # Examine new data
    cmd = "res = clf.predict( np.c_["
    for i in range( N_feas ): cmd += "feas2[%d],"%i
    cmd = cmd[:-1]+" ] )"
    exec(cmd)

    acc = 100*sum(res)/len(res)
    print("%f%%"%acc)


    # Plottings
    try:
        plt.close("all")
    except ValueError:
        print("No figures to close.")

    # Time series
    fig = plt.figure()
    f1 = fig.add_subplot(111)
    #plot_parse( par, f1 )
    plot_ai( par, f1 )
    #plot_data( data, f1 )

    # Scatter plot
    fig = plt.figure()
    f2 = fig.add_subplot(111)
    ft = np.transpose(feas1)
    fbgt = np.transpose(feas_bg)
    f2.scatter(ft[0],ft[1],c='r',label=name1)
    f2.scatter(feas2[0],feas2[1],label=wu.get_name(name2))
    #feas_bg = np.transpose(feas_bg)
    #f2.scatter(feas_bg[0],feas_bg[1],label='Mock')
    #f2.scatter(feas_bg[0],feas_bg[1],alpha=0.)

    plt.legend()
    f2.set_xlabel('Feature 1')
    f2.set_ylabel('Feature 2')

    # Plot decision boundary of the original data
    if N_feas==2:
        fac2 = 5.
        xx,yy = np.meshgrid(np.arange(dim_minmax[0][0],dim_minmax[0][1],h[0]/fac/fac2),\
                            np.arange(dim_minmax[1][0],dim_minmax[1][1],h[1]/fac/fac2))
        db = clf.predict(np.c_[xx.ravel(),yy.ravel()])
        db = db.reshape(xx.shape)
        f2.contourf(xx,yy,db,alpha=0.3)

    #return [ft,xx,yy,db,data,feas2,par.t_parsed_avg,par.data_parsed_avg,par.sig_parsed_avg,acc]



