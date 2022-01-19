import crepe
from scipy.io import wavfile
import numpy as np
import math
from pprint import pprint
import matplotlib.pyplot as plt
from midiutil.MidiFile import MIDIFile
from Levenshtein import distance as levenshtein_distance
from Levenshtein import ratio as levenshtein_ratio
from sklearn.cluster import AffinityPropagation
import pickle
import random
import time
import subprocess
from pathlib import Path
from scipy.interpolate import CubicSpline
from scipy.interpolate import splev
import os

# step_size=5
# filename='anger_1-28_0006'
# sr, audio = wavfile.read('../speech_data/emovdb-angry/'+filename+'.wav')
# time_, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True, step_size=step_size)

defaultMinConfidence = 0.7

class Object(object):
    pass

# calls CREPE to compute time, frequency and confidence for an audio file
def getPrediction(sr, audio, stepSize=5):
	time, frequency, confidence, _ = crepe.predict(audio, sr, viterbi=True, step_size=stepSize)
	return time, frequency, confidence

def getNotes(filename,verterbi=True,stepSize=5):
	return 0

# converts a single frequency into its corresponding musical note according to MIDI standard (returns a float!)
def convert_to_midi_note(f):
	return 12*math.log(f/440,2)+69

# converts a list of frequencies into a list of musical note pitches according to MIDI standard (returns floats!)
def audio_to_notes(frequency):
	return list(map(convert_to_midi_note,frequency))

# returns an interpolated value based on an index (float) and a list of values
def getInterpolatedY(values, index):
	check_int = isinstance(index, int)
	if(check_int):
		# index is integer and return value can directly be read from the list of values
		return values[index]
	else:
		# index is float and return value needs to be calculated from 2 neighboring values
		low = values[math.floor(index)]
		high = values[math.ceil(index)]
		return low + (high-low)*(index%1)

# returns max sampling of raw audio data as described in Section 3.4.3 of the thesis
def maxSampling(audio_data, num_samples):
	stepSize = len(audio_data)/num_samples
	# initialize result as zeroes
	result = [0 for x in range(num_samples)]
	for i in range(num_samples):
		# low is the lower boundary of the current interval
		low = i*stepSize
		# high is the upper boundary of the current interval
		high = min((i+1)*stepSize,len(audio_data)-1)
		# midval refers to the highest value within the current interval
		midVal = np.max(audio_data[math.ceil(low):math.floor(high)])
		# value at the lower end of the interval
		lowVal = getInterpolatedY(audio_data,low)
		# value at the upper end of the interval
		highVal = getInterpolatedY(audio_data,high)
		# compute maximum of all values
		val = np.max([lowVal,midVal,highVal])		
		result[i]=val
	return result

def getSegments(notes, time, confidence, minConfidence = defaultMinConfidence):
	data = list(zip(time,notes))
	if(minConfidence == 0):
		segments=[notes]
	else:
		segments=[]
		high=False
		currentSegment=[]
		for i,x in enumerate(confidence):
			if x<minConfidence:
				if high:
					high=False
					segments.append(currentSegment[:])
					currentSegment=[]
				continue
			else:
				if high==False:
					high=True
				currentSegment.append(list(data[i]))
	return segments


def segmentsToSimpleNotes(segments):
	notes = []
	for segment in segments:
		m=np.mean(np.array(segment).T[1])
		pitch=round(m)
		notes.append(pitch)
	return notes

def notesToDerivativeString(word):
	result=""
	for n in range(len(word)-1):
		result = result + chr(word[n+1]-word[n]+79)
	return (result)

def notesToTernaryDerivativeString(word):
	result=""
	for n in range(len(word)-1):
		if(word[n+1]>word[n]):
			# U stands for up (positive derivative)
			result = result + 'U'
		elif(word[n+1]==word[n]):
			# N stands for no change (derivative=0)
			result = result + 'N'
		else:
			# D stands for down (negative derivative)
			result = result + 'D'
	return result


def notesToRoundedString(notes, t, c, stepSize, minConf = defaultMinConfidence):
	resultstr = ""
	resultlist = []
	for i in range(math.ceil(len(notes)/stepSize)):
		conf = np.mean(c[i*stepSize : min(len(notes),(i+1)*stepSize)])
		pitch = round(np.mean(notes[i*stepSize : min(len(notes),(i+1)*stepSize)]))
		if(conf > minConf):
			resultlist.append(pitch)
			resultstr = resultstr + chr(pitch+15)
		else:
			resultlist.append(0)
			resultstr = resultstr + "~"
	
	return resultstr.strip('~')
	

def notesToString(notes):
	result = ""
	for note in notes:
		result = result + chr(note)
	return result

# convert a list of segments to midi notes (approach #1+2)
def segmentsToMidiNotes(segments, time, stepSize = 10, volume = None, average = True):
	_notes = []
	if(volume == None):
		volume = np.ones(len(time))
	for seg in segments:
		if(average == True):
			if(len(seg)>0):
				m=np.mean(np.array(seg).T[1])
				pitch=round(m)
				_time=seg[0][0]*250
				duration=stepSize*len(seg)/4
				#mf.addNote(track, channel, pitch, time, duration, volume)
				
				
				_note = Object()
				_note.pitch = pitch
				_note.time = _time
				_note.duration = duration
				_note.volume = 100
				_notes.append(_note)
		else:
			for n in seg:
				m=n[1]
				pitch=round(m)
				_time=n[0]*250
				duration=stepSize/4
				_note = Object()
				_note.pitch = pitch
				_note.time = _time
				_note.duration = duration
				_note.volume = 100
				_notes.append(_note)
		
	return _notes

# save notes as a midi file (approach #3)
def saveMidi(notes, filename, stepSize = 5, volumes = []):
	mf=MIDIFile(1)
	mf.addProgramChange(0, 0, 0, 26)
	track=0
	mf.addTrackName(track, 0, "Sample Track")
	mf.addTempo(track,0,60)
	channel=0
	count=0
	#gap = -5
	last=0
	for i in range(len(notes)):
		if len(volumes)>0:
			if volumes[i]>0.05:
				volume=100+round(volumes[i]*100)
			else:
				volume=0
		else:
			volume=100
		# *0.001 to convert from ms to beats (60 bpm = 1s per beat)
		t = stepSize*0.001*i
		dur = stepSize*0.001
		# interpolation can sometimes produce unrealistic pitches. This sets their volume to 0
		pitch=round(notes[i])
		if pitch<10:
			volume=0
			pitch=10
		elif pitch>80:
			volume=0
			pitch=80
		else:
			volume=100
		if (pitch != last):
			mf.addNote(track,channel,pitch,t,dur,volume)
			last = pitch
			count=count+1
	
	# add one silent note for nicer cutoff at the end
	mf.addNote(track,channel,50,stepSize*0.001*len(notes),5*stepSize*0.001,0)
		
	print(f"added {count} notes. Total length: {t+5*stepSize*0.001}s")
	print(f"Writing MIDI to {filename}")
	with open(filename, 'wb') as outf:
		mf.writeFile(outf)
		outf.close()

# convert midi file to mp3 file
def saveAsMP3(input_file,output_file):
	# location of sound font
	sound_font = "F:/Masterarbeit/Python/fluidsynth/weedsgm3.sf2"
	# location of compiled fluidsynth executable
	fluidsynth_path="F:/Masterarbeit/Python/fluidsynth/bin/fluidsynth.exe"
	# sample rate used for the output file
	sample_rate=44100
	# execute command line command for fluidsynth midi conversion
	subprocess.call([fluidsynth_path, '-ni', sound_font, input_file, '-F', output_file, '-r', str(sample_rate)])
	print('output:')
	print(output_file)


def saveMidiFile(notes, filename, stepSize = 10, enable_confidence = True):
	mf=MIDIFile(1)
	mf.addProgramChange(0, 0, 0, 17)
	track=0
	time=0
	mf.addTrackName(track, time, "Sample Track")
	mf.addTempo(track,time,60*1000/stepSize)
	channel=0
	volume=100
	skip=1
	for note in notes:
		mf.addNote(track,channel,note.pitch,note.time,note.duration,note.volume)
	mf.addNote(track, channel, 1, notes[len(notes)-1].time+notes[len(notes)-1].duration,50,0)
	_filename="dump/"+filename+"_"+str(stepSize)+"ms.mid"
	print(f"Writing MIDI to {_filename}")
	with open(_filename, 'wb') as outf:
		mf.writeFile(outf)

# distance function used for approach #1
def modified_levenshtein_distance(x,y):
	minX = np.min([ord(c) for c in x])
	minY = np.min([ord(c) for c in y])
	if(minX>minY):
		_min = minY
		_max = np.max([ord(c) for c in x])
	else:
		_min = minX
		_max = np.max([ord(c) for c in y])
	minDist = 100
	for shift in range(_max-_min+1):
		if(minX>minY):
			y_mod = "".join([chr(ord(c)+shift) for c in y])
			l = levenshtein_distance(x,y_mod)
		else:
			x_mod = "".join([chr(ord(c)+shift) for c in x])
			l = levenshtein_distance(y,x_mod)
		minDist = min(minDist,l)
	return minDist

# distance function used for approach #3
def modified_notes_distance(notes1,notes2):
	if(len(notes1)<len(notes2)):
		notes_shorter = notes1
		notes_longer = notes2
	else:
		notes_shorter = notes2
		notes_longer = notes1
	
	min_dif = 1000
	for i in range(len(notes_longer)-len(notes_shorter)):
		dif = 0
		for j in range(len(notes_shorter)):
			dif = dif + abs(notes_shorter[j]-notes_longer[i+j])
		min_dif=min(dif,min_dif)
	min_dif_norm = min_dif / len(notes_shorter)
	return min_dif_norm

# perform clustering on numerical encodings of pitch patterns (approach #3)
# calculating the distance matrices is rather time consuming, so the option to load previously computed data was added
def getNoteClusters(notes,notes_deriv,labels,loadData=False,saveData=False,saveTxtDistances=False):
	if loadData:
		# read distance data from local files
		similarity_deriv = readList("similarity_deriv")
		similarity_direct = readList("similarity_direct")
		similarity = similarity_direct + similarity_deriv
	else:
		# compute distances (this can take a while)
		similarity_direct = np.array([[modified_notes_distance(n1,n2) for n1 in notes] for n2 in notes])
		print('direct similarity calculation complete')
		similarity_deriv = np.array([[modified_notes_distance(n1,n2) for n1 in notes_deriv] for n2 in notes_deriv])
		print('derivative similarity calculation complete')
		similarity = similarity_direct + similarity_deriv
	if saveData:
		# write distance data to local files
		saveList(similarity,"similarity")
		saveList(similarity_deriv,"similarity_deriv")
		saveList(similarity_direct,"similarity_direct")
	if saveTxtDistances:
		# write json-formatted distance data to local files (for usage with the cluster visualization tool)
		printArray(similarity,"distances.txt","float")
		printArray(similarity_direct,"distances_direct.txt","float")
		printArray(similarity_deriv,"distances_deriv.txt","float")
		print('similarity data written to disk')
	affprop = AffinityPropagation(affinity="precomputed", damping=0.9)
	# negative squared distance in accordance with the documentation
	affprop.fit(-similarity*similarity)
	if affprop.labels_[0] == -1:
		print('did not converge')
		return affprop
	_notes = np.asarray(notes)
	_labels = np.asarray(labels)
	exemplars=[]
	clusters=[]
	for cluster_id in np.unique(affprop.labels_):
		exemplar = labels[affprop.cluster_centers_indices_[cluster_id]]
		cluster = [w for w in np.unique(_labels[np.nonzero(affprop.labels_==cluster_id)])]
		exemplar_index=affprop.cluster_centers_indices_[cluster_id]
		cluster_indices = [w for w in np.unique(np.nonzero(affprop.labels_==cluster_id))]
		exemplars.append(exemplar_index)
		clusters.append(cluster_indices)
		cluster_str = ", ".join(cluster)
		print("Exemplar: %s Elements: %s" % (exemplar, cluster_str))
	
	print(exemplars)
	print(clusters)
	return affprop

# perform clustering on string encodings of pitch patterns (approach #1+2)
# results are printed to the console but can also be accessed through the affprop object that is returned
def getStringClusters(_words, labels, mode):
	#So that indexing with a list will work
	words = np.asarray(_words)
	if mode == "normal":
		# use modified levenshtein distance
		similarity = -1*np.array([[modified_levenshtein_distance(w1,w2) for w1 in words] for w2 in words])
	elif mode == "ratio":
		# use unmodified levenshtein ratio
		similarity = -1*np.array([[levenshtein_ratio(w1,w2) for w1 in words] for w2 in words])
	# save distances to file for visualization
	printArray(similarity,"dist_string","float")
	affprop = AffinityPropagation(affinity="precomputed", damping=0.5)
	# run affinity propagation clustering
	affprop.fit(similarity)
	if affprop.labels_[0] == -1:
		print('did not converge')
		return affprop
	_labels = np.asarray(labels)
	exemplars=[]
	clusters=[]
	for cluster_id in np.unique(affprop.labels_):
		exemplar = labels[affprop.cluster_centers_indices_[cluster_id]]
		cluster = [w for w in np.unique(_labels[np.nonzero(affprop.labels_==cluster_id)])]
		exemplar_index=affprop.cluster_centers_indices_[cluster_id]
		cluster_indices = [w for w in np.unique(np.nonzero(affprop.labels_==cluster_id))]
		exemplars.append(exemplar_index)
		clusters.append(cluster_indices)
		cluster_str = ", ".join(cluster)
		print("Exemplar: %s Elements: %s" % (exemplar, cluster_str))
	
	print(exemplars)
	print(clusters)
	return affprop

def interpolateNotes(notes,c,t,minConfidence,samplingInterval):
	# filter notes based on confidence threshold
	notes_filtered = [note_i for note_i, conf_i in zip(notes, c) if conf_i>minConfidence]
	# filter times based on confidence threshold
	times_filtered = [time_i for note_i, conf_i, time_i in zip(notes, c, t) if conf_i>minConfidence]
	# inverse of times_filtered
	times_invalid = [time_i for note_i, conf_i, time_i in zip(notes, c, t) if conf_i<=minConfidence]
	# generate spline function from anchor points
	cs = CubicSpline(times_filtered, notes_filtered)

	begin=times_filtered[0]
	end=times_filtered[len(times_filtered)-1]
	# list of points to be sampled
	xs = np.arange(begin,end,samplingInterval)
	# sampled pitch value
	notes_spl = cs(xs)
	# sampled derivative values
	notes_deriv = cs(xs,1)*0.005*25

	return notes_spl, notes_deriv

def interpolateNotesList(notes,c,t,minConfidence = defaultMinConfidence,samplingInterval = 0.05):
	notes_splined=[]
	notes_splined_deriv=[]
	for i in range(len(notes)):
		notes_spl,notes_deriv = interpolateNotes(notes[i],c[i],t[i],minConfidence,samplingInterval)
		# add relevant values to return objects
		notes_splined.append(notes_spl)
		notes_splined_deriv.append(notes_deriv)
	
	return notes_splined,notes_splined_deriv

# save an array to a file in standard json format
def printArray(arr,filename,datatype):
	if datatype=="float":
		strings = [','.join(["%.2f" % number for number in _arr]) for _arr in arr]
	else:
		strings = [','.join([str(element) for element in _arr]) for _arr in arr]
	text ='[' + '],['.join(strings) + ']'
	print('writing array')
	print(filename)
	with open(filename, "w") as text_file:
		text_file.write(text)
	return

# auxiliary function for string clustering
def wordToLabel(word, words, labels):
	return labels[words.index(word)]

# reads a file and returns the results of the pitch detection through CREPE
def processFile(directory, filename, stepSize = 5):
	sr, audio = wavfile.read(directory+"/"+filename)
	t,f,c = getPrediction(sr,audio,stepSize=stepSize)
	notes = audio_to_notes(f)
	return notes, t, c

# read files from a set of directories and speakers and return all pitch prediction results (this can take a while)
# optional: save all data locally
def processBatchData(directories, speakers, saveLocally = False, stepSize = 5, minConfidence = defaultMinConfidence):
	notes = []
	files = []
	times = []
	conf = []
	for directory in directories:
		for speaker in speakers:
			count = 0
			path = directory+speaker
			for filename in os.listdir(path):
				print(count)
				f = os.path.join(path, filename)
				# checking if it is a file
				if os.path.isfile(f):
					_notes, t, c = processFile(path, filename, stepSize, minConfidence = minConfidence)
					notes.append(_notes)
					files.append(filename)
					times.append(t)
					conf.append(c)
					count = count + 1
	
	if saveLocally:
		saveList(notes,"notes")
		saveList(files,"files")
		saveList(times,"times")
		saveList(conf,"conf")

	return notes, files, times, conf

# save a python list to a file
def saveList(list, targetFile="dump"):
	with open(targetFile,"wb") as fp:
		pickle.dump(list, fp)

# read a python list from a file
def readList(sourceFile="dump"):
	with open(sourceFile,"rb") as fp:
		res = pickle.load(fp)
		return res
