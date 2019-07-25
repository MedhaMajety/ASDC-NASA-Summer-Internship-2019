#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:59:04 2019

@author: mmajety
"""



import numpy as np
import pandas as pd
import os.path
import time
import json
import csv
from pyhdf.SD import SD, SDC
from pydap.client import open_url
import pydap.lib
import os
from scipy.stats.stats import pearsonr
import re
from astropy.time import Time
import matplotlib.pyplot as plt
import statistics

pydap.lib.CACHE = "/tmp/pydap-cache/"
####
#create an ascii data file containing CALIPSO LID_L2_PSC retrieval results
data_out='CorrelationValues.json'
path_data_out='/Users/mmajety/Documents/ASDC/subsets/PSCMASK/'
#dictionary keys: granulename,time,lat,lon,fcf_fetch_time,psc_dim0,volcanic_ash_dim0,sulfate_sub_dim0,elevated_smoke_dim0,psc_indices
path='/Users/mmajety/Documents/ASDC/subsets/PSCMASK/'
pathpsc='/Users/mmajety/Documents/ASDC/subsets/PSCMASK/'
pathvfm = '/Users/mmajety/Documents/ASDC/subsets/VFM/2019/01/'
#sample granule: /Users/wbaskin/git/test-CALIPSO/data/CAL_LID_L2_PSCMask-Prov-V1-00.2011-01-04T00-00-00ZN.hdf  

debug=False

df_all=pd.DataFrame(columns=['Time','CompFlag1','CompFlag2','CompFlag3','CompFlag4','CompFlag5','CompFlag6','CompFlag7'])


def TAItoUTC(tai_time_value):
    #Convert float in TAI93 (since 1JAN93 epoch) to UTC in isot format
    
    t93 = Time('1993-01-01T00:00:00.000', format='isot', scale='utc')
    epoch93 = t93.gps
    tai93time = Time(tai_time_value+epoch93, format='gps', scale="utc")
    utc_time=tai93time.isot
    return utc_time

#specific implementation for CAL_LID_L2_PSCMask (default sds_name)
#Read an HDF4 SDS dataset of type String and return an list of Strings
#hdf = SD(FILE_NAME, SDC.READ)
#sds_name is the name of the SDS parameter to process
def hdf4_SDS_String_to_list(hdf,sds_name='L1_Input_Filenames'):
    sds_char_array = hdf.select(sds_name)[:]
    data_array=[]        
    iterable=range(sds_char_array.shape[0])
    for dim0_index in iterable:
        data_array.append(sds_char_array[dim0_index].tostring().decode('utf-8'))
    return data_array

#default implementation: replace CAL_LID_L1*. prefix with CAL_LID_L2_VFM-Standard-V4-20.
def change_calipso_granulename_prefix(string_array,prefix_regex='CAL_LID_L1[^\.]*.',new_prefix='CAL_LID_L2_VFM-Standard-V4-20.'):
    data_array=[]
    for string_value in string_array:
        data_array.append(re.sub(prefix_regex, new_prefix, string_value))     
    return data_array


def fetch_HDF_data_local(granulename, param_names):
    dict_out={}
    hdf = SD(granulename, SDC.READ)
    for param_name in param_names:
        #print('reading datasetname: '+param_name)
        values=np.array(hdf.select(param_name)[:])
        dict_out[param_name]=values
    return dict_out

def delete_dictionary_key(dict_in,keys):
    #create a copy to prep for Pandas Dataframe Creation
    dict_out=dict_in.copy()
    for key in keys:
        del dict_out[key]
    return dict_out
    

#Joins  CAL_LID_L2_PSCMASK and CAL_LID_L2_VFM dataframes using the. 
#Profile_Time column as the key
#
#see: Joining key columns on an index in url: 
#https://pandas-docs.github.io/pandas-docs-travis/user_guide/merging.html 
    
def join_psc_vfm_coverage(pscmask_df,vfm_df):
    #pscmask_df.join(vfm_df, on='Profile_Time')
    #df_merged=pd.merge(pscmask_df,vfm_df, left_on='Profile_Time',  how='pscmask_df', sort=False)
    #0.024
    df_merged=pd.merge(pscmask_df,vfm_df,on='Profile_Time',how='inner')
    return df_merged

#Create matching orbit coverage using intersection of Profile_Time values from
#a CAL_LID_L2_PSCMASK granule (orbit subset) and a CAL_LID_L2_VFM product 
def match_coverage(psc_df,vfm_dict):
    #vfm_dict_temp=delete_dictionary_key(vfm_dict,['psc_dim0','fcf_fetch_time','granulename','psc_count5km','pscmask_orbit_index'])
    vfm_dict_temp={'Profile_Time':np.concatenate(vfm_dict['Profile_Time']),'vfm_flag':vfm_dict['flag_array']}
    vfm_df=pd.DataFrame(vfm_dict_temp)
    #vfm_dict_temp.clear() #this may be overkill...
    df_out=join_psc_vfm_coverage(psc_df,vfm_df)
    return df_out
    '''
    epsilon=0.001
    start_times=np.array([psc_df['Profile_Time'].values[0],vfm_dict['Profile_Time'][0][0]])
    stop_times=np.array([psc_df['Profile_Time'].iloc[-1],vfm_dict['Profile_Time'][-1][0]])
    #print('start_times= '+TAItoUTC(start_times[0])+' '+TAItoUTC(start_times[1]),'stop_times= '+TAItoUTC(stop_times[0])+' '+TAItoUTC(stop_times[1]) )
    if abs(start_times[0]-start_times[1])>epsilon :
        start_time_filter=start_times.max()
        if start_times[0]>start_times[1]:
            time_indices=np.where(vfm_dict['Profile_Time']>=start_time_filter)[0]
            vfm_dict['Profile_Time']=vfm_dict['Profile_Time'][time_indices]
            vfm_dict['Latitude']=vfm_dict['Latitude'][time_indices]
            vfm_dict['Longitude']=vfm_dict['Longitude'][time_indices]
            vfm_dict['flag_array']= vfm_dict['flag_array'][time_indices]
            print('changed vfm_dict. new starttime= '+str(start_time_filter))
        else:
            psc_df=pd.DataFrame(psc_df.loc[psc_df['Profile_Time']>=start_time_filter, ['Orbit_Index', 'Latitude', 'Longitude', 'Profile_Time','compflag1','compflag2','compflag3','compflag4','compflag5','compflag6']])

            #psc_df=psc_df.DataFrame(psc_df['Profile_Time']>=start_time_filter, ['Orbit_Index', 'Latitude', 'Longitude', 'Profile_Time','compflag1','compflag2','compflag3','compflag4','compflag5','compflag6']])
            #psc_df=psc_df.DataFrame((psc_df['Profile_Time']>=start_time_filter) & (psc_df['Profile_Time']<=stop_time_filter) , ['Orbit_Index', 'Latitude', 'Longitude', 'Profile_Time','compflag1','compflag2','compflag3','compflag4','compflag5','compflag6']])

            print('changed psc_df. new starttime= '+str(start_time_filter),psc_df['Profile_Time'][0])
    if abs(stop_times[0]-stop_times[1])>epsilon:
        stop_time_filter=stop_times.min()
        if stop_times[0]<stop_times[1]:
            time_indices=np.where(vfm_dict['Profile_Time']<=stop_time_filter)[0]
            vfm_dict['Profile_Time']=vfm_dict['Profile_Time'][time_indices]
            vfm_dict['Latitude']=vfm_dict['Latitude'][time_indices]
            vfm_dict['Longitude']=vfm_dict['Longitude'][time_indices]            
            vfm_dict['flag_array']= vfm_dict['flag_array'][time_indices]
            print('changed vfm_dict. new stoptime= '+str(stop_time_filter))
        else:
            psc_df=pd.DataFrame(psc_df.loc[psc_df['Profile_Time']<=stop_time_filter, ['Orbit_Index', 'Latitude', 'Longitude', 'Profile_Time','compflag1','compflag2','compflag3','compflag4','compflag5','compflag6']])
            print('changed psc_df. new stoptime= '+str(stop_time_filter),psc_df['Profile_Time'][psc_df.index[-1]])
    return psc_df,vfm_dict
    '''
    

def get_VFM_PSC_dictionary(granulename, pscmask_orbit_index, url_path='https://opendap.larc.nasa.gov:443/opendap/CALIPSO/LID_L2_VFM-Standard-V4-20/2011/01/',northern_hemisphere=True,lat_filter=50.00000,opendap=True):
    dict_out={}
    url=url_path+granulename
    
    start1 = time.time()
    od_Feature_Classification_Flags=None
    dataset = None
    if opendap:
        print("accessing opendap url: "+url)
        dataset = open_url(url)
    else:
        granulename_long = pathvfm+granulename
        #print("reading local file: "+granulename_long)
        dataset=fetch_HDF_data_local(granulename_long,['Feature_Classification_Flags'])
        
    od_Feature_Classification_Flags=dataset['Feature_Classification_Flags'][:,0:1165]
        
    ##################################
    #faster implementation. Uses subsetted slices instead of the entire Feature_Classification_Flags array.
    feature_type_id=4  #feature_type_id=4 is stratospheric Aerosol
    #volcanic_ash_mask    = np.uint16(0b0010010000000100) #confident,volcanic ash
    feature_type_mask    = np.uint16(0b0000000000000111)
    feature_subtype_mask = np.uint16(0b0000111000000000)
    od_Feature_Classification_Flags=np.asarray(od_Feature_Classification_Flags)

    feature_type=od_Feature_Classification_Flags&feature_type_mask
    strat_aerosol_type_indices=np.where(feature_type==feature_type_id)
    if len(strat_aerosol_type_indices)>0:
        strat_dim0_indices=np.unique(strat_aerosol_type_indices[0])
        
        #subset feature_type array to process only the dim0 slices that contain Stratospheric Aerosol Feature Type 
        feature_type=feature_type[strat_dim0_indices,:]
        
        # zero out all values that are not stratospheric aerosols 
        ft_masked=np.ma.masked_where(feature_type!=feature_type_id, feature_type)
        
        #apply mask of stratospheric aerosol feature type to subsetted version of the Feature_Classification_Flags variable
        fcf_masked = np.ma.masked_where(ft_masked.mask, od_Feature_Classification_Flags[strat_dim0_indices,:])
        
        feature_subtypes=fcf_masked&feature_subtype_mask
        #shift the bits of the Feature_Classification_Flags value so that bits 10-12 are now bits 1-3
        feature_subtypes=feature_subtypes>>9
              
        #extract indices corresponding to each of the stratospheric aerosol subtypes
        psc_indices=np.where(feature_subtypes==1)
        
        if len(psc_indices)>0:
            #volcanic_ash_sub_indices=np.where(feature_subtypes==2)
            #sulfate_sub_indices=np.where(feature_subtypes==3)
            #elevated_smoke_sub_indices=np.where(feature_subtypes==4)
            #volcanic_ash_dim0=np.unique(np.asarray(volcanic_ash_sub_indices)[0])
            #sulfate_sub_dim0=np.unique(np.asarray(sulfate_sub_indices)[0])
            #elevated_smoke_dim0=np.unique(np.asarray(elevated_smoke_sub_indices)[0])
            psc_dim0=np.unique(np.asarray(psc_indices)[0])

            if not opendap:
                dataset=fetch_HDF_data_local(pathvfm+granulename,['Latitude','Longitude','Profile_Time'])
                
            lat=dataset['Latitude'][:]
            lon=dataset['Longitude'][:]   
            profile_time=dataset['Profile_Time'][:]         

            end1 = time.time()
            fcf_fetch_time=end1-start1
            
            #note: for winter months in arctic get data above 50deg North latitude
            subset_indices_50deg=np.where(lat>=lat_filter)[0]
            if not northern_hemisphere:
                 subset_indices_50deg=np.where(lat<=lat_filter)[0]
            
            lat_subset=lat[subset_indices_50deg]
            lon_subset=lon[subset_indices_50deg]
            profile_time_subset=profile_time[subset_indices_50deg]           
            flag_array = np.zeros((lat_subset.shape[0]), dtype=int)
            np.put(flag_array,psc_dim0,1)
            dict_out={'granulename':granulename,'fcf_fetch_time':fcf_fetch_time,'psc_count5km':len(psc_dim0),'Profile_Time':profile_time_subset,'Latitude':lat_subset,'Longitude':lon_subset,'psc_dim0':psc_dim0,'flag_array':flag_array,'pscmask_orbit_index':pscmask_orbit_index}#,'volcanic_ash_dim0':volcanic_ash_dim0,'sulfate_sub_dim0':sulfate_sub_dim0,'elevated_smoke_dim0':elevated_smoke_dim0,'psc_indices':psc_indices}

        return dict_out
    
def gen_psc_mask_dict(longnamepsc):      
        #start1 = time.time()
        
        #create a Pandas dataframe containing Polar Stratospheric Cloud Composition data
        #and corresponding orbit and position values
        FILE_NAME = os.path.abspath(longnamepsc)
        print("processing: "+ FILE_NAME)
        hdfpsc = SD(FILE_NAME, SDC.READ)
        psc=np.array(hdfpsc.select('PSC_Composition')[:])
        Orbit_numberpsc = np.array(hdfpsc.select('Orbit_Index')[:]) 
        Latpsc = np.array(hdfpsc.select('Latitude')[:])
        Lonpsc = np.array(hdfpsc.select('Longitude')[:])
        profile_timepsc=np.array(hdfpsc.select('Profile_Time')[:])
        trop_height=np.array(hdfpsc.select('Tropopause_Height')[:])
        psccompmodify = psc[:,0]
        
        #Get the list of source L1 granulenames and convert to a list of L2_VFM granulenames
        psc_lid_l1_source_filenames=hdf4_SDS_String_to_list(hdfpsc)
          
        data = {'source_filenames':psc_lid_l1_source_filenames,'PSCComp':psccompmodify, 'Orbit_Index':Orbit_numberpsc, 'Latitude':Latpsc, 'Longitude': Lonpsc, 'Profile_Time':profile_timepsc, 'Tropopause_Height':trop_height}


        #add six columns to dataframe corresponding to the six PSC_Composition types
        #1 indicates that at least one of the PSC_Composition type was encountered
        #in the vertical column above the corresponding 5km (lon-lat) location
        
        #Comp_flag = 0	No Cloud detected
        #Comp_flag = 1	Liquid Supercooled Ternary (sulfuric acid, water, nitric acid) Solution (STS) droplets
        #Comp_flag = 2	Mix1: STS + low number densities/volumes of Nitric Acid Trihydrate (NAT) particles
        #Comp_flag = 3	Mix2: STS + intermediate number densities/volumes of NAT particles
        #Comp_flag = 4	Water ice clouds
        #Comp_flag = 5	Mix2-enhanced: STS + high number densities/volumes of NAT particles
        #Comp_flag = 6	Wave ice: Mountain wave induced water ice clouds (R > 50)
        
        iterable= [1,2,3,4,5,6] #range(1,7)
        for compflag in iterable:
         
            psc_comp_id=np.where(psc==compflag)
            psccomp1_dic = dict()
            for y,x in np.asarray(psc_comp_id).transpose():
                if(y in psccomp1_dic):
                    psccomp1_dic[y].append(x)
                else:
                    psccomp1_dic[y] = [x]
                    
            keys=list(psccomp1_dic.keys())
            flag_array = np.zeros((psc.shape[0]), dtype=int)
            np.put(flag_array,keys,1)
            data['compflag'+str(compflag)] = flag_array.tolist()
            
        #special case: compflag7 (any flag 1-6)  
        psc_comp_id=np.where(psc>0)
        psccomp1_dic = dict()
        for y,x in np.asarray(psc_comp_id).transpose():
            if(y in psccomp1_dic):
                psccomp1_dic[y].append(x)
            else:
                psccomp1_dic[y] = [x]
                
        keys=list(psccomp1_dic.keys())
        flag_array = np.zeros((psc.shape[0]), dtype=int)
        np.put(flag_array,keys,1)     
        data['compflag7'] = flag_array.tolist()  
        '''           
        #special case: compflag7 (any flag 1-6)  
        psc_comp_id=np.where(psc>0)
        psccomp1_dic = dict()
        for y,x in np.asarray(psc_comp_id).transpose():
            if(y in psccomp1_dic):
                psccomp1_dic[y].append(x)
            else:
                psccomp1_dic[y] = [x]
                
        keys=list(psccomp1_dic.keys())
        flag_array = np.zeros((psc.shape[0]), dtype=int)
        np.put(flag_array,keys,1)
        data['compflag7'+str(compflag)] = flag_array.tolist()           
        '''           
            
            
            
        #end1=time.time()
                
        return(data)

#split dictionary containing relevant pscmask data into array of Pandas Dataframes
#each row in the returned array contains a dictionary containing the following entries
# 'CAL_LID_L1_granulename','orbit_index_filter','PSC_Dataframe'
def split_pscmask_to_orbits(data_dict):
    list_out = []
    source_filenames = data_dict['source_filenames']
    #create a temporary copy to prep for Pandas Dataframe Creation
    temp_data_dict=data_dict.copy()
    del temp_data_dict['source_filenames']
    
    df=pd.DataFrame(temp_data_dict)  
    temp_data_dict.clear()
    
    for orbit_index_filter in range(len(source_filenames)):
        df_psc=pd.DataFrame(df.loc[df['Orbit_Index'] == orbit_index_filter, ['Orbit_Index', 'Latitude', 'Longitude', 'Profile_Time','Tropopause_Height','compflag1','compflag2','compflag3','compflag4','compflag5','compflag6','compflag7']])
        list_out.append({'CAL_LID_L1_granulename':source_filenames[orbit_index_filter],'orbit_index_filter':orbit_index_filter,'PSC_Dataframe':df_psc})
    
    return list_out
        
    
with open(pathpsc+'granulenames_origpsc.txt', 'r') as f:
    for line in f:
        fnamepsc = line.rstrip('\n')
        if fnamepsc=='':
            continue
    
        longnamepsc=pathpsc+fnamepsc
        psc_dict = gen_psc_mask_dict(longnamepsc)
        psc_dict_list=split_pscmask_to_orbits(psc_dict)
 
        psc_vfm_filenames=change_calipso_granulename_prefix(psc_dict['source_filenames'])
        
       
        #indices in psc_vfm_filenames correspond to Orbit_Index in dataframe: df

        for psc_granule_dict in psc_dict_list:
            
            index=psc_granule_dict['orbit_index_filter']
            vfm_granulename=psc_vfm_filenames[index]
            vfm_granulename=re.sub('\.hdf','_Subset.hdf',vfm_granulename)
            ###if debug==True, limit loop to first 3 VFM orbit files
            if debug:
                if index !=4:
                    print('DEBUG skipping: '+vfm_granulename)
                    continue
            #################################################
            print("processing: "+pathvfm+vfm_granulename,'INDEX: '+str(index))
            psc_df=psc_granule_dict['PSC_Dataframe']
            vfm_dict=get_VFM_PSC_dictionary(vfm_granulename,index,lat_filter=psc_df['Latitude'].values[-1],opendap=False)
            psc_count5km=vfm_dict['psc_count5km']
            print("number of indices with psc flag: ",psc_count5km)

            if psc_count5km>1:
                lat=np.array(vfm_dict['Latitude'])
                lon=np.array(vfm_dict['Longitude'])
                profile_time=np.array(vfm_dict['Profile_Time'])
                
                
                
                print('psc_lat_size:',len(psc_df['Latitude']),'vmf_lat_size:',len(lat), 'Start_Time dif in seconds (psc-vfm): '+str(psc_df['Profile_Time'].values[0]-profile_time[0]),'Stop_Time dif (psc-vfm): '+str(psc_df['Latitude'].values[-1]-lat[-1]))
                merged_df=match_coverage(psc_df,vfm_dict)
                #print('psc_lat_size:',len(psc_df['Latitude']),'vmf_lat_size:',len(lat), 'Start_Time dif in seconds (psc-vfm): '+str(psc_df['Profile_Time'].values[0]-profile_time[0]),'Stop_Time dif (psc-vfm): '+str(psc_df['Latitude'].values[-1]-lat[-1]))

                
                flag_array=np.array(vfm_dict['flag_array'])


                
                #Results for day: 2011-01-04

                #corr1 0.0525
                #corr2 0.3286
                #corr3 0.2928
                #corr4 0.2179
                #corr5 0.1392
                
                '''
                corr1, _ = pearsonr(flag_array, np.array(psc_df['compflag1'].values[:]))                
                corr2, _ = pearsonr(flag_array, np.array(psc_df['compflag2'].values[:]))
                corr3, _ = pearsonr(flag_array, np.array(psc_df['compflag3'].values[:]))
                corr4, _ = pearsonr(flag_array, np.array(psc_df['compflag4'].values[:]))
                corr5, _ = pearsonr(flag_array, np.array(psc_df['compflag5'].values[:]))
                corr6, _ = pearsonr(flag_array, np.array(psc_df['compflag6'].values[:]))
                corr7, _ = pearsonr(flag_array, np.array(psc_df['compflag7'].values[:]))
                '''

                corr1, _ = pearsonr(np.array(merged_df['vfm_flag'].values[:]), np.array(merged_df['compflag1'].values[:]))                
                corr2, _ = pearsonr(np.array(merged_df['vfm_flag'].values[:]), np.array(merged_df['compflag2'].values[:]))
                corr3, _ = pearsonr(np.array(merged_df['vfm_flag'].values[:]), np.array(merged_df['compflag3'].values[:]))
                corr4, _ = pearsonr(np.array(merged_df['vfm_flag'].values[:]), np.array(merged_df['compflag4'].values[:]))
                corr5, _ = pearsonr(np.array(merged_df['vfm_flag'].values[:]), np.array(merged_df['compflag5'].values[:]))
                corr6, _ = pearsonr(np.array(merged_df['vfm_flag'].values[:]), np.array(merged_df['compflag6'].values[:]))
                corr7, _ = pearsonr(np.array(merged_df['vfm_flag'].values[:]), np.array(merged_df['compflag7'].values[:]))
                print('Pearsons correlations:',corr1,corr2,corr3,corr4,corr5,corr6,corr7)
                dict_corr={'Granule_Name':vfm_granulename, 'CompFlag1':corr1, 'CompFlag2':corr2, 'CompFlag3':corr3, 'CompFlag4':corr4, 'CompFlag5':corr5, 'CompFlag6':corr6,'CompFlag7':corr7}
                dict_plot={'Time': merged_df['Profile_Time'].values[0], 'CompFlag1':abs(corr1), 'CompFlag2':abs(corr2), 'CompFlag3':abs(corr3), 'CompFlag4':abs(corr4), 'CompFlag5':abs(corr5), 'CompFlag6':abs(corr6),'CompFlag7':abs(corr7)}
                dfplot = pd.DataFrame([dict_plot])
                dfplot = dfplot.fillna(0)
                df_all = df_all.append(dfplot)
                df_all = df_all.fillna(0)
                psclog = open(data_out,'a')
                #print(dict_out,file=psclog)
                jsonstring = json.dump(dict_corr,psclog)
                print(file=psclog)
                psclog.close()
print('CompFlag1 Mean', statistics.mean(df_all['CompFlag1']))
print('CompFlag2 Mean', statistics.mean(df_all['CompFlag2']))
print('CompFlag3 Mean', statistics.mean(df_all['CompFlag3']))
print('CompFlag4 Mean', statistics.mean(df_all['CompFlag4']))
print('CompFlag5 Mean', statistics.mean(df_all['CompFlag5']))
print('CompFlag6 Mean', statistics.mean(df_all['CompFlag6']))
print('CompFlag7 Mean', statistics.mean(df_all['CompFlag7']))

print('CompFlag1 sd', statistics.stdev(df_all['CompFlag1']))
print('CompFlag2 sd', statistics.stdev(df_all['CompFlag2']))
print('CompFlag3 sd', statistics.stdev(df_all['CompFlag3']))
print('CompFlag4 sd', statistics.stdev(df_all['CompFlag4']))
print('CompFlag5 sd', statistics.stdev(df_all['CompFlag5']))
print('CompFlag6 sd', statistics.stdev(df_all['CompFlag6']))
print('CompFlag7 sd', statistics.stdev(df_all['CompFlag7']))

print('CompFlag1 min', min(df_all['CompFlag1']))
print('CompFlag2 min', min(df_all['CompFlag2']))
print('CompFlag3 min', min(df_all['CompFlag3']))
print('CompFlag4 min', min(df_all['CompFlag4']))
print('CompFlag5 min', min(df_all['CompFlag5']))
print('CompFlag6 min', min(df_all['CompFlag6']))
print('CompFlag7 min', min(df_all['CompFlag7']))

print('CompFlag1 max', max(df_all['CompFlag1']))
print('CompFlag2 max', max(df_all['CompFlag2']))
print('CompFlag3 max', max(df_all['CompFlag3']))
print('CompFlag4 max', max(df_all['CompFlag4']))
print('CompFlag5 max', max(df_all['CompFlag5']))
print('CompFlag6 max', max(df_all['CompFlag6']))
print('CompFlag7 max', max(df_all['CompFlag7']))
ax1 = df_all.plot.scatter(x='Time',y='CompFlag1')
plt.show()            

ax1 = df_all.plot.scatter(x='Time',y='CompFlag2')
plt.show()            

ax1 = df_all.plot.scatter(x='Time',y='CompFlag3')
plt.show()            

ax1 = df_all.plot.scatter(x='Time',y='CompFlag4')
plt.show()            

ax1 = df_all.plot.scatter(x='Time',y='CompFlag5')
plt.show()            
  
ax1 = df_all.plot.scatter(x='Time',y='CompFlag6')
plt.show()            

ax1 = df_all.plot.scatter(x='Time',y='CompFlag7')
plt.show()            

fig7, ax7 = plt.subplots()
ax7.set_title('Multiple Samples with Different sizes')
ax7.boxplot(df_all)
plt.show()
'''
pd.options.display.mpl_style = 'default'
df_all.boxplot()
plt.show(block=True)
'''
    #TEST READABILITY OF THE DATA DICTIONARY IN THE LOG FILE...   
    
data = []
with open(data_out) as f:
    for line in f:
        data.append(json.loads(line)) 
        

        '''
        df_comp_1 = pd.DataFrame(df.loc[df['PSCComp'] == 1, ['PSCComp', 'Orbit_Index', 'Latitude', 'Longitude', 'Profile_Time']])
        ...
        df_comp_6 = pd.DataFrame(df.loc[df['PSCComp'] == 6, ['PSCComp', 'Orbit_Index', 'Latitude', 'Longitude', 'Profile_Time']])
    
    
        df_orbit_0 = pd.DataFrame(df.loc[df['Orbit_Index'] == 0, ['Orbit_Index', 'Latitude', 'Longitude', 'Profile_Time']])
        ...
        df_orbit_13 = pd.DataFrame(df.loc[df['Orbit_Index'] == 13, ['Orbit_Index', 'Latitude', 'Longitude', 'Profile_Time']])
        '''
