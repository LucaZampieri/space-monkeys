from astropy.table import Table
import matplotlib.pyplot as plt
from collections import OrderedDict
import seaborn as sns

datadir = '/home/ole/Desktop/Ole/Uni/plasticc/data'
metafilename = datadir + '/training_set_metadata.csv'
metadata = Table.read(metafilename, format='csv')

lc_filename = datadir + '/training_set.csv'
data = Table.read(lc_filename, format='csv')

filters = {
    0 : 'u',
    1 : 'g',
    2 : 'r',
    3 : 'i',
    4 : 'z',
    5 : 'y'
    }

def get_obsids(data,depth):
    obsids=[]
    last = 0
    for i in data['object_id'][:depth]:
        if data['object_id'][i]!=last:
            obsids.append(data['object_id'][i])
        last = data['object_id'][i]
    return obsids

### PARAMETERS ###
passband = 1
##################

#print('Data:\n',data.info('stats'))
#print('Metadata:\n',metadata.info('stats'))

#metaobsid = get_obsids(metadata,5000)
#print(metaobsid)

target = 42
mask_2 = (metadata['target']==target)
#obsids = get_obsids(data,5000)

obsids = metadata[mask_2]['object_id']
print(obsids)
#fig, axs = plt.subplots(1, len(obsids), figsize=(5, 5), sharey=True)
for num, obsid in enumerate(obsids):
    
    for passband, color in zip([0,1,2,3],['r+','b+','g+','m+']):
        mask = (data['object_id']==obsid) & (data['passband']==passband)
        plt.errorbar(data[mask]['mjd'],data[mask]['flux'],xerr=None,yerr=data[mask]['flux_err'],
                     fmt=color,markeredgewidth=2,capthick=2,elinewidth=2,label=str(passband))

    # Find target number in metadata
    md_mask = (metadata['object_id']==obsid)
    target = metadata[md_mask]['target'][0]
    #axs[num].set_title('%s'%target)
    
    
    plt.title('target=%s' % target)
    plt.xlabel('MJD')
    plt.ylabel('Flux')
    plt.legend()

    plt.show()
