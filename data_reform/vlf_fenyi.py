from astropy.io import fits
import pytz 
from datetime import datetime
def filename2header(filename):
    file_suffix = '.cos' # file extension
    headerlist = filename.split(',')
    hdr = fits.Header()
    hdr['ins'] = ','.join(headerlist[0:3]) # instrument
    hdr['freq'] = float(headerlist[3]) # frequency
    hdr['f_unit']='Hz' # unit of frequency
    hdr['sp_rate'] =float(headerlist[4]) # sample rate 
    hdr['sp_unit']='Hz' # unit of sample rate
    hdr['int'] = headerlist[5] 
    hdr['ldate'] =headerlist[6][:8] # local date 
    hdr['ltime'] = headerlist[6][9:15]# local time 
    yyyy=headerlist[6][:4] 
    mon=headerlist[6][4:6]
    date=headerlist[6][6:8]
    hh=headerlist[6][9:11]
    mm=headerlist[6][11:13]
    ss=headerlist[6][13:15]    
    # Ref: https://kite.com/python/answers/how-to-convert-local-datetime-to-utc-in-python
    timezone=pytz.timezone('Etc/GMT+8') # time zone
    time_str=yyyy+'-'+mon+'-'+date+' '+hh+':'+mm+':'+ss #  time string
    naive_datetime=datetime.strptime(time_str,"%Y-%m-%d %H:%M:%S") # local time
    local_datetime = timezone.localize(naive_datetime, is_dst=None) 
    utc_datetime = local_datetime.astimezone(pytz.utc) # conver into UTC
    hdr['tzone']='UTC+8' # time zone
    hdr['UTC']=utc_datetime.strftime("%Y-%m-%d UT %H:%M:%S") #  UTC format
    hdr['lonlat'] = ','.join(headerlist[7:9]) #  remove the '.cos'
    return hdr

if __name__ == '__main__':
    filename = 'EWNS,Trig,fenyi_bb,1437,250000.00,10s_10s,20200229_075900,27.91N,114.70E.cos'
    hdr = filename2header(filename)
    print(type(hdr['ins']))
#    print(hdr['ins'])
#    print(hdr['lonlat'])
    print(repr(hdr))

