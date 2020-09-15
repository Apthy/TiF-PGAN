import librosa
import numpy as np
import os
#from strechableNumpyArray import StrechableNumpyArray
import heapq
from numba import njit
import h5py
import matplotlib.pyplot as plt


#dir = 'out/Data/test/30sec/'
#datasetpath = dir+'1/1024P.hdf5'
#with h5py.File(datasetpath, 'r') as hf:
#    try:
#        keys = list(hf.keys())
#        hf.close()
#    except:
#        print('error')
@njit
def pghi(spectrogram, tgrad, fgrad, a, M, L, tol=10):
    spectrogram = spectrogram.copy()
    abstol = -20
    phase = np.zeros_like(spectrogram)
    max_val = np.amax(spectrogram)  # Find maximum value to start integration
    max_x, max_y = np.where(spectrogram == max_val)
    max_pos = max_x[0], max_y[0]

    if max_val <= abstol:  # Avoid integrating the phase for the spectogram of a silent signal
        print('Empty spectrogram')
        return phase

    M2 = spectrogram.shape[0]
    N = spectrogram.shape[1]
    b = L / M

    sampToRadConst = 2.0 * np.pi / L  # Rescale the derivs to rad with step 1 in both directions
    tgradw = a * tgrad * sampToRadConst
    fgradw = - b * (fgrad + np.arange(
        spectrogram.shape[1]) * a) * sampToRadConst  # also convert relative to freqinv convention

    magnitude_heap = [(-max_val, max_pos)]  # Numba requires heap to be initialized with content
    spectrogram[max_pos] = abstol

    small_x, small_y = np.where(spectrogram < max_val - tol)
    for x, y in zip(small_x, small_y):
        spectrogram[x, y] = abstol  # Do not integrate over silence

    while max_val > abstol:
        while len(magnitude_heap) > 0:  # Integrate around maximum value until reaching silence
            max_val, max_pos = heapq.heappop(magnitude_heap)

            col = max_pos[0]
            row = max_pos[1]

            # Spread to 4 direct neighbors
            N_pos = col + 1, row
            S_pos = col - 1, row
            E_pos = col, row + 1
            W_pos = col, row - 1

            if max_pos[0] < M2 - 1 and spectrogram[N_pos] > abstol:
                phase[N_pos] = phase[max_pos] + (fgradw[max_pos] + fgradw[N_pos]) / 2
                heapq.heappush(magnitude_heap, (-spectrogram[N_pos], N_pos))
                spectrogram[N_pos] = abstol

            if max_pos[0] > 0 and spectrogram[S_pos] > abstol:
                phase[S_pos] = phase[max_pos] - (fgradw[max_pos] + fgradw[S_pos]) / 2
                heapq.heappush(magnitude_heap, (-spectrogram[S_pos], S_pos))
                spectrogram[S_pos] = abstol

            if max_pos[1] < N - 1 and spectrogram[E_pos] > abstol:
                phase[E_pos] = phase[max_pos] + (tgradw[max_pos] + tgradw[E_pos]) / 2
                heapq.heappush(magnitude_heap, (-spectrogram[E_pos], E_pos))
                spectrogram[E_pos] = abstol

            if max_pos[1] > 0 and spectrogram[W_pos] > abstol:
                phase[W_pos] = phase[max_pos] - (tgradw[max_pos] + tgradw[W_pos]) / 2
                heapq.heappush(magnitude_heap, (-spectrogram[W_pos], W_pos))
                spectrogram[W_pos] = abstol

        max_val = np.amax(spectrogram)  # Find new maximum value to start integration
        max_x, max_y = np.where(spectrogram == max_val)
        max_pos = max_x[0], max_y[0]
        heapq.heappush(magnitude_heap, (-max_val, max_pos))
        spectrogram[max_pos] = abstol
    return phase

#%%

pathToBaseDatasetFolder = 'Data/'
folderNames =  ['test']# , 'test', 'valid']['sc09']
dirs = [pathToBaseDatasetFolder + folderName for folderName in folderNames]
audios = []
i = 0
total = 0

#the files we currently have are all of different sizes so we need to find the average size
lengths = []
names = []
fns = []
counter = 0
maxsr = 0
minsr = 9999999
srs = []
le=1
#this has been added as a preprocessing step as it filters all of the files that are too la/home/apthy/PycharmProjects/newSimplePganrge out of the dataset
# and stores the remenents in an array ready to be fully loaded
for directory in dirs:
    print(directory)
    for file_name in os.listdir(directory):
        if True:#counter<100: #load 250 files max
            if file_name.endswith('.wav'):
                audio, sr = librosa.load(directory + '/' + file_name, sr=None, dtype=np.float64)
                le = int(1024*1024/2)
                #print(len(audio))
                if(len(audio)>=le):#if it is longer than will make a nice sized image then add it
                    srs.append(sr)
                    if(sr>44100):

                        print('[critical] sr too large: ',file_name)
                    elif (sr < 44100):
                        if(sr<minsr):
                            minsr = sr
                        print('[CRITICAL error] sr too small: ', file_name)
                    lengths.append(len(audio))
                    fns.append(directory + '/' + file_name)
                    names.append(file_name)
                    counter+=1
                else:
                    print('[Error] File Too Short: ',file_name)
lengths = np.array(lengths)
fns = np.array(fns)
names = np.array(names)
mapper = np.vstack((lengths, fns,names))
plt.plot(lengths)
plt.title('the Lengths Of Files')
plt.plot([0,len(lengths)],[np.mean(lengths),np.mean(lengths)],color='m')#meanline
plt.plot([0, len(lengths)], [np.mean(lengths)+np.sqrt(np.var(lengths)), np.mean(lengths)+np.sqrt(np.var(lengths))],color='g')  # pos std
plt.plot([0, len(lengths)], [np.mean(lengths)-np.sqrt(np.var(lengths)), np.mean(lengths)-np.sqrt(np.var(lengths))],color='r')  # neg std
plt.show()
lengths = np.sort(lengths)#makes a nicer visual representation of the length split
srs.sort()
plt.plot(srs)
plt.title('the sample rates Of Files')
plt.show()
maxlens=np.mean(lengths)+np.sqrt(np.var(lengths)) #this was chosen as the files are all random lengths, assumed to be roughly gaussian
minlens = np.mean(lengths)/4# - np.sqrt(np.var(lengths)) this was not appropriate as there were lots of smaller files that would not fill the entire image.
plt.plot(lengths)
plt.plot([0,len(lengths)],[np.mean(lengths),np.mean(lengths)],color='m')#meanline
plt.plot([0, len(lengths)], [np.mean(lengths)+np.sqrt(np.var(lengths)), np.mean(lengths)+np.sqrt(np.var(lengths))],color='g')  # pos std
plt.plot([0, len(lengths)], [np.mean(lengths)-np.sqrt(np.var(lengths)), np.mean(lengths)-np.sqrt(np.var(lengths))],color='r')  # neg std
plt.show()
maxes = 44100*round(round(.8*maxlens)/44100)#16000 this was the default as only 1 second clips were used
mins = int(44100*round(minlens)/44100)
plt.plot(lengths)
plt.title('the Lengths Of Files sorted')
plt.plot([0,len(lengths)],[np.mean(lengths),np.mean(lengths)],color='m')#meanline
plt.plot([0, len(lengths)], [maxes, maxes],color='g')  # pos std
plt.plot([0, len(lengths)], [mins, mins],color='r')  # neg std
plt.show()
print("done!")
#lengths = []
#fns = []
#names = []
##filter the elements that are too large out
#for item in mapper.T:
#    #print(item[0])
#    if ((int(item[0])<=maxes)&(int(item[0])>=mins)):
#        lengths.append(item[0])
#        fns.append(item[1])
#        names.append(item[2])
mapper = np.vstack((lengths, fns))
names = [os.path.splitext(x)[0] for x in names]

from ourLTFATStft import LTFATStft
import ltfatpy
from old.modGabPhaseGrad import modgabphasegrad
ltfatpy.gabphasegrad = modgabphasegrad  # The original function is not implemented for one sided stfts on ltfatpy
fft_hop_size = 64*8
fft_window_length = 256*8
num = 1
dir = 'out/Data/test/30sec/'
while os.path.exists(dir+str(num)):
    num+=1
os.makedirs(dir+str(num))


clipBelow = -10
#anStftWrapper = LTFATStft()
gs = {'name': 'gauss', 'M': fft_window_length}
#mainfile = h5py.File(pathToBaseDatasetFolder+'testdataset.h5py','a',)
#mainfile.create_dataset("mydataset", (len(fns),), dtype='i')




for fileNamePath in fns:#for each file in the list
    #gc.collect()
    anStftWrapper = LTFATStft()
    audio, sr = librosa.load(fileNamePath, sr=None, dtype=np.float64)#load the audio and its sample rate

    audio = audio[:le]
    L = len(audio)
    #if len(audio) < maxes:         #square and pad with 0
    #    before = int(np.floor((maxes - len(audio)) / 2))
    #    after = int(np.ceil((maxes - len(audio)) / 2))
    #    audio = np.pad(audio, (before, after), 'constant', constant_values=(0, 0))
    if np.sum(np.absolute(audio)) < len(audio) * 1e-4:
        print(fileNamePath, "doesn't meet the minimum amplitude requirement")
        continue
    #make the spectogram and its derrivs
    realDGT = anStftWrapper.oneSidedStft(signal=audio, windowLength=fft_window_length, hopSize=fft_hop_size)
    spectrogram = anStftWrapper.logMagFromRealDGT(realDGT, clipBelow=np.e ** clipBelow, normalize=True)
    spectrogramshrink = np.delete(spectrogram,0,1)
    spectrogramshrink = np.delete(spectrogramshrink, 0,0)
    print(spectrogram.shape)
    tgradreal, fgradreal = ltfatpy.gabphasegrad('abs', np.abs(realDGT), gs, fft_hop_size)
    #tshrink = np.delete(tgradreal,0,1)
    #fshrink = np.delete(fgradreal,0,1)
    #tshrink = np.delete(tshrink, 0, 0)
    #fshrink = np.delete(fshrink, 0, 0)
    #fig = plt.figure(frameon=False, figsize=[10,280])
    #ax = plt.Axes(fig, [0., 0., 1., 1.])
    #ax.set_axis_off()
    #fig.add_axes(ax)
    #ax.imshow(spectrogram, aspect='auto')

    #mpl.rcParams['image.cmap'] = 'inferno'
    #plt.figure(figsize=(15, 2.5))
    #plt.subplot(131)
    #ltfatpy.plotdgtreal(spectrogram, fft_hop_size, fft_window_length, dynrange=40)
    #plt.title('Log magnitude')
    #plt.subplot(132)
    #ltfatpy.plotdgtreal(tgradreal * (np.abs(spectrogram) > np.exp(-5)), fft_hop_size, fft_window_length, normalization='lin')
    #plt.title('Phase time-derivative')
    #plt.subplot(133)
    #ltfatpy.plotdgtreal(fgradreal * (np.abs(spectrogram) > np.exp(-5)), fft_hop_size, fft_window_length, normalization='lin')
    #plt.title('Demodulated phase frequency-derivative')
    #plt.show()

    #toSave = np.array([spectrogram,tgradreal,fgradreal])
    toSave = np.dstack((spectrogram,tgradreal,fgradreal))
    #np.save(dir+str(num)+'/'+str(i)+' sr'+str(sr) + ' hop'+str(fft_hop_size)+' winLen'+str(fft_window_length)+'SpecsAndDerrivs'+str(names[i])+'.npy', toSave)

    with h5py.File(dir+str(num)+'/large.hdf5', 'a') as hf:
        try:
            hf.create_dataset(str(names[i]), data=toSave,compression=None)
            hf.close()
        except:
            print('no space')
            hf.close()



    #print(h5py.File(dir+str(num)+'/large.hdf5', 'r').keys())
    #fig.savefig(f"{pre}{outdir}{indir}{filename}.jpg")
    #tgrads[index] = tgradreal
    #fgrads[index] = fgradreal

    #spectrogram[0,:] = -10
    #tgradreal[0,:] = -10
    #fgradreal[0,:] = -10
    #spectrogram[:, 0] = -10
    #tgradreal[:, 0] = -10
    #fgradreal[:, 0] = -10
    #PGHI to reconstruct the audio
    #phase = pghi(spectrogram, tgradreal, fgradreal, fft_hop_size, fft_window_length, L, tol=8)
    #anStftWrapper.reconstructSignalFromLoggedSpectogram(spectrogram, phase, windowLength=fft_window_length,hopSize=fft_hop_size)
    #phase = pghi(spectrogram, tgradreal, fgradreal, fft_hop_size, fft_window_length, L, tol=8)
    #anStftWrapper.reconstructSignalFromLoggedSpectogram(spectrogram, phase, windowLength=fft_window_length,
    #                                                    hopSize=fft_hop_size)
    #from scipy.io.wavfile import write
    #print("Audio conversion complete file output at: "+ dir+str(num)+'/'+str(i))
    #write(dir+str(num)+'/'+str(i)+' sr'+str(sr) + ' hop'+str(fft_hop_size)+' winLen'+str(fft_window_length)+'SpecsAndDerrivs'+str(names[i])+'.wav', sr, audio)#sr is maybe wrong
    #print(i)
    i += 1
    print(i,'/',len(fns),' Files completed',round(100*(i/len(fns)),3),'%')
    #print(len(audio))
    #if i > 100:
    #    i -= 100
    #    total += 100
    #    print("Just loaded 1000 files! The total now is:", total)
print("Finished! I loaded", i, "audio files.")
