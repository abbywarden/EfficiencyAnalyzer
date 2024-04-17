import os 
import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import re 

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='takes a csv file from a run and outputs problematic chambers/vfats to json files',
                    epilog='execution : python3 VFAT_Efficiency_Analyzer.py --input_csv input_csv.csv --output_dir /path/to/jsons')
                   
parser.add_argument('-icsv', '--input_csv', type=str , help="csv of the run to be used",required=True)
parser.add_argument('-v', '--verbose',
                    action='store_true')  # on/off flag
parser.add_argument("-o",'--output_dir', type=str, help="Output dir to store json files",required=False, default="output_jsons")

args = parser.parse_args()

#create folder and copy files 
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(args.input_csv, index_col=0, dtype=int)

run_number = re.findall("(\d+)_RPCMonitor", args.input_csv)[0]

#calculate efficiency per vfat 
df["vfat_eff"] = df["matchedRecHit"]/df["propHit"]
#get the eta partition for the vfat
df["eta_partition"] = df["VFAT"] % 8 
eta_to_vfat = {
    0:[0,8,16], 
    1:[1,9,17], 
    2:[2,10,18],
    3:[3,11,19], 
    4:[4,12,20], 
    5:[5,13,21], 
    6:[6,14,22], 
    7:[7,15,23]
}

################################## functions ##################################
def replace_partition(eta):
    return eta_to_vfat.get(eta,eta)

#iqr function 
def find_outliers(df, col_name, thresh) :
    q1 = df[col_name].quantile(0.25)
    q3 = df[col_name].quantile(0.75)
    iqr = q3-q1 
    outlier_value = q1 - thresh*iqr
    outliers = df[(df[col_name] < outlier_value)].copy()
    return outliers, outlier_value

#drop chosen outliers from the dataframe --> this handles cases when the dataframes have different columns (but same Region,Chamber,Layer,VFAT)
def drop_outliers(df_main, df_outliers) :
    df_output = df_main.merge(df_outliers, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    return df_output

#slightly different from above -- to drop duplicates between two dataframes
def drop_duplicates(df_main, df_compare) :
    df_main = df_main[['Region', 'Chamber', 'Layer', 'VFAT']] #slim down the df
    df_compare = df_compare[['Region', 'Chamber', 'Layer', 'VFAT']] #slim down the df 
    df_output = pd.merge(df_main, df_compare, how='outer', indicator=True)
    df_output = df_output.loc[df_output._merge == 'left_only', ['Region', 'Chamber', 'Layer', 'VFAT']]
    return df_output

#this is to get the average ch eff given a low efficiency vfat or eta partition 
def get_status(avg_ch_eff, type) :
    return "%s w/ low eff compared to avg ch eff : %f" % (type, avg_ch_eff)

#note : mean() does not include the NaN values but can have cases when have matchedRechit = 0, propHit != 0 
#so first remove these
oddball_vfats = df.loc[(df['matchedRecHit'] == 0) & (df['propHit'] != 0)]
df = df.loc[(df['matchedRecHit'] > 0) & (df['propHit'] != 0)] #this also removes masked vfats & chambers --> will mess with the bad apple calc. re-evaluate? 

df[['avg_ch_eff', 'avg_ch_propHit']] = df.groupby(['Region', 'Chamber', 'Layer'], as_index=True)[['vfat_eff', 'propHit']].transform(lambda x: x.mean()) #calculate avg chamber eff; propHit 
df[['ch_eff_std', 'ch_propHit_std']] = df.groupby(['Region', 'Chamber', 'Layer'], as_index=True)[['vfat_eff', 'propHit']].transform(lambda x: x.std()) #calculate std chamber eff; propHit

chamber_df = df[['Region', 'Chamber','Layer', 'avg_ch_eff', 'avg_ch_propHit']].drop_duplicates().reset_index(drop=True) ##by chamber -> to find the bad chambers 


################################## chambers ##################################
#find chamber outliers w.r.t the entire system

if args.verbose : 
    print("opening %s"%args.input_csv)

#determine low stats chambers; 
lowstats_ch, chpropHit = find_outliers(chamber_df, 'avg_ch_propHit', 1.5)  
ch_df_slim = drop_outliers(chamber_df, lowstats_ch) #drop before considering low eff chambers 

loweff_ch, cheff = find_outliers(ch_df_slim, 'avg_ch_eff', 1.5)
## drop chamber outliers from dataframe before moving on to calculate average vfat eff, average eta partition eff, average eta partition propHit
df_slim_temp = drop_outliers(df, lowstats_ch)
df_slim = drop_outliers(df_slim_temp, loweff_ch) 

lowstats_ch['Status'] = pd.Series(['avg chamber propHit < %f'%chpropHit for x in range(len(lowstats_ch.index))])
loweff_ch['Status'] = pd.Series(['avg chamber eff < %f'%cheff for x in range(len(loweff_ch.index))])

lowstats_ch["Status"] = 'avg chamber propHit < %f'%chpropHit
loweff_ch["Status"] = 'avg chamber eff < %f'%cheff

if args.verbose : 
    print("For Run : %s, Average Chamber PropHit = %f, and Average Chamber Efficiency = %f"%(run_number, chpropHit, cheff))
    print("Found %i chambers with average chamber prophit < %f and %i chambers with average chamber efficiency < %f"%(lowstats_ch.shape[0], chpropHit, loweff_ch.shape[0], cheff))
 
#combine
bad_ch = pd.concat([lowstats_ch,loweff_ch]).reset_index(drop=True)
bad_ch = bad_ch.sort_values(by = ['Region', 'Chamber', 'Layer'])

#put into csv
bad_ch.to_csv(str(output_dir) + "/problematic_chambers_%s.csv"%run_number, sep=";", index=False)
if args.verbose :
    print("Output for flagged chambers stored here : %s" %str(output_dir) + "/problematic_chambers_%s.csv" %run_number)
################################ indiv. vfats ################################
#find low stat vfats w.r.t chamber avg propHit
lowstats_vfat, vfatpropHit = find_outliers(df_slim, 'propHit', 1.5)
df_slim = drop_outliers(df_slim, lowstats_vfat)  

#find vfat outliers w.r.t the entire system 
outlier_vfat, outlier_vfateff = find_outliers(df_slim, 'vfat_eff', 1.5)
df_slim[['outlier_vfat']] = (df_slim['vfat_eff'] < outlier_vfateff) 
          
if args.verbose :
    print("Now looking for vfats (and eta partitions) in GE11 with an efficiency < %f"%outlier_vfateff)

# bad_apple == 1 (only one outlier vfat in eta partition) 
# bad_apple == 2 (2/3 vfats are outliers in eta partition)
# bad_apple == 3 (ie a bad_branch --> all vfats in eta partition are outliers
# true & true & true = 3  # True & True & false = 2  # true & false & false = 1  # false & false & false = 0 
df_slim[['bad_apple']] = df_slim.groupby(['Region', 'Chamber', 'Layer', 'eta_partition'], as_index=False)['outlier_vfat'].transform(sum) 

#round 1 harvesting : get the immediate bad apples and bad barrels 
badapple_vfats = df_slim.loc[(df_slim['bad_apple'] != 3) & (df_slim['outlier_vfat'] == True)].copy()
badbarrel_vfats = df_slim.loc[df_slim['bad_apple'] == 3].copy()

if args.verbose :
    print("Found %i outlier vfats within GE11 and %i outlier eta partitions" % (badapple_vfats.shape[0], badbarrel_vfats.shape[0]))

# drop the bad vfats from the dataframe;
df_slim_2 = drop_outliers(df_slim, badapple_vfats)  
df_slim_2 = drop_outliers(df_slim_2, badbarrel_vfats) 


#round two : now look more closely at vfats by comparing to the average chamber efficiency (after removal of all above)

if args.verbose :
    print("Now looking for vfats with low efficiency compared to the given average chamber efficiency") 

#first need to redo all the average calculations & vfat outliers
df_slim_2 = df_slim_2.drop(['outlier_vfat', 'bad_apple'], axis = 1)
df_slim_2[['avg_ch_eff', 'avg_ch_propHit']] = df_slim_2.groupby(['Region', 'Chamber', 'Layer'], as_index=True)[['vfat_eff', 'propHit']].transform(lambda x: x.mean())
df_slim_2[['ch_eff_std', 'ch_propHit_std']] = df_slim_2.groupby(['Region', 'Chamber', 'Layer'], as_index=True)[['vfat_eff', 'propHit']].transform(lambda x: x.std())
##time to recalculate the outlier vfats 
df_slim_2[['avg_etapart_eff', 'avg_etapart_propHit']] = df_slim_2.groupby(['Region', 'Chamber', 'Layer', 'avg_ch_eff', 'eta_partition'], as_index=False)[['vfat_eff', 'propHit']].transform(lambda x: x.mean())
#1.5 times std from mean
df_slim_2[['outlier_eta']] = df_slim_2['avg_etapart_eff'] < ( df_slim_2['avg_ch_eff'] - 1.5*df_slim_2['ch_eff_std'])

#round 2 harvesting : get the bad apples and bad barrel (now comparing to the average chamber eff)
badbarrel2_vfats = df_slim_2.loc[(df_slim_2['outlier_eta'] == True)].copy() #will need to make sure to recombine vfats that were pulled at earlier stages 

#drop the bad eta partitions before moving on 
df_slim_3 = drop_outliers(df_slim_2, badbarrel2_vfats) 

#recalculate outlier_vfat, avg_ch_eff again
df_slim_3[['avg_ch_eff', 'avg_ch_propHit']] = df_slim_3.groupby(['Region', 'Chamber', 'Layer'], as_index=True)[['vfat_eff', 'propHit']].transform(lambda x: x.mean())
df_slim_3[['ch_eff_std', 'ch_propHit_std']] = df_slim_3.groupby(['Region', 'Chamber', 'Layer'], as_index=True)[['vfat_eff', 'propHit']].transform(lambda x: x.std())
df_slim_3[['outlier_vfat']] = df_slim_3['vfat_eff'] < ( df_slim_3['avg_ch_eff'] - 1.5*df_slim_3['ch_eff_std'])

badapple2_vfats = df_slim_3.loc[(df_slim_3['outlier_vfat'] == True)]

if args.verbose :
    print("Found %i outlier vfats within individual chambers and %i outlier eta partitions" % (badapple2_vfats.shape[0], badbarrel2_vfats.shape[0]))

##############################################################################

#formatting for putting info into jsons 

#converting eta partitions back to individual vfats while keeping eta's together -> have yet to figure out how to do below without these calculations that are not used 
byeta_barrel = badbarrel_vfats.groupby(['Region', 'Chamber', 'Layer', 'avg_ch_eff', 'eta_partition'], as_index=False)[['vfat_eff', 'propHit']].mean().rename(columns={'vfat_eff':'avg_etapart_eff', 'propHit':'avg_etapart_propHit'}) ##by etapartition
byeta_barrel2 = badbarrel2_vfats.groupby(['Region', 'Chamber', 'Layer', 'avg_ch_eff', 'eta_partition'], as_index=False)[['vfat_eff', 'propHit']].mean().rename(columns={'vfat_eff':'avg_etapart_eff', 'propHit':'avg_etapart_propHit'}) ##by etapartition

#replace eta partitions by vfats 
byeta_barrel['VFAT'] = byeta_barrel['eta_partition'].apply(replace_partition)
byeta_barrel2['VFAT'] = byeta_barrel2['eta_partition'].apply(replace_partition)

byeta = byeta_barrel.groupby(['Region', 'Chamber', 'Layer'])['VFAT'].apply(list).reset_index()
byeta2 = byeta_barrel2.groupby(['Region', 'Chamber', 'Layer', "avg_ch_eff"])['VFAT'].apply(list).reset_index()

#if want to remove duplicates from : low eff vfats(round 1 ) & eta partitions (round 2) 
#also to remove duplicates from : low stats vfats (round 1) & eta partitions (round 2)
#this removal favors issue w/ eta partition 
eta_byvfat2 = byeta2.explode('VFAT').explode('VFAT')
badapple_vfats = drop_duplicates(badapple_vfats, eta_byvfat2)
lowstats = drop_duplicates(lowstats_vfat, eta_byvfat2)

oddball_vfats = oddball_vfats.groupby(['Region', 'Chamber', 'Layer'])['VFAT'].apply(list).reset_index() 
lowstats_vfats = lowstats_vfat.groupby(['Region', 'Chamber', 'Layer'])['VFAT'].apply(list).reset_index() 
badapple_vfats = badapple_vfats.groupby(['Region', 'Chamber', 'Layer'])['VFAT'].apply(list).reset_index() 
badapple2_vfats = badapple2_vfats.groupby(['Region', 'Chamber', 'Layer', 'avg_ch_eff'])['VFAT'].apply(list).reset_index() 

oddball_vfats["Status"] = "vfats w/ matchedpropHit = 0 but prophit !=0"
lowstats_vfats["Status"] = "vfats w/ low propHit compared to GE1/1: propHit < %f" % vfatpropHit
badapple_vfats["Status"] = "vfats w/ low eff compared to GE1/1 : eff < %f"% (outlier_vfateff)
badapple2_vfats["Status"] = badapple2_vfats["avg_ch_eff"].apply(get_status, type = "vfats")
badapple2_vfats = badapple2_vfats.drop(['avg_ch_eff'], axis = 1)
byeta["Status"] = "all vfats in eta partition w/ low eff compared to GE1/1 : eff < %f"%(outlier_vfateff)
byeta2["Status"] = byeta2["avg_ch_eff"].apply(get_status, type = "eta partition")
byeta2 = byeta2.drop(['avg_ch_eff'], axis = 1)


bad_vfats = pd.concat([oddball_vfats, lowstats_vfats, badapple_vfats, byeta, badapple2_vfats, byeta2]).sort_values(by = ['Region', 'Chamber', 'Layer']).reset_index(drop=True)
bad_vfats.to_csv(str(output_dir) + "/bad_vfats_%s.csv"%run_number, sep=";", index=False)
if args.verbose :
    print("Output for flagged vfats stored here : %s" % str(output_dir) + "/bad_vfats_%s.csv" % run_number)
##############################################################################
