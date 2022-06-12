from cgi import print_arguments
from cv2 import CV_16U, imshow
from matplotlib.image import imsave
import numpy as np
import cv2
import matplotlib.pyplot as plt
# from picamera import PiCamera
from time import sleep 
plt.rcParams['figure.figsize'] = [16, 9]


# camera = PiCamera(resolution = (1920,1080))
# camera.brightness=2
# camera.contrast=100
# camera.shutter_speed = 60000
# camera.start_preview()
# sleep(20)
# camera.start_recording('test.h264')
# camera.wait_recording(10) 
# camera.stop_recording()
# camera.stop_preview()
# camera.close()


def generateROIDetectionTemplate(rowHeight, columnHeight, numberOfRows, numberOfRepetitions):
    template_g = np.repeat(np.array([[[0, 255, 0]]]), rowHeight, axis=0)
    template_r = np.repeat(np.array([[[0, 0, 255]]]), rowHeight, axis=0)
    template_b = np.repeat(np.array([[[255, 0, 0]]]), rowHeight, axis=0)
    template_n = np.repeat(np.array([[[0, 0, 0]]]), rowHeight, axis=0)
    templates = np.array([template_g,template_r,template_b,template_n
                            ])
    template_concat = templates[0]
    for i in range(1, numberOfRepetitions*numberOfRows):
        j = i % 4
        template_concat = np.concatenate((template_concat,templates[j]),axis=0)

    template = np.repeat(template_concat, columnHeight, axis=1)
    
    return template.astype(np.float32)/255.0

def generateSyncTemplate(rowHeight):
    template_g = np.repeat(255, rowHeight, axis=0)
    template_n = np.repeat(0, rowHeight, axis=0)
    template = np.concatenate((template_g,template_g,template_n,template_g,template_n),axis=0)
    return template.astype(np.float32)/255.0



video = cv2.VideoCapture('testV2lineasGordas.h264')
status, frame = video.read(0)

frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

rowHeight = 8
columnHeight = 20
numberOfRepetitions = 6

detection_template = (generateROIDetectionTemplate(rowHeight = rowHeight, 
                                        columnHeight = columnHeight, 
                                        numberOfRows = 4, 
                                        numberOfRepetitions = numberOfRepetitions)*255).astype(np.uint8)
#fig, ax = plt.subplots(1,3, sharey=True)
##ax[0].imshow(detection_template)
##ax[0].set_title("Detection template")

numberOfRepetitions = 1
calibration_template = (generateROIDetectionTemplate(rowHeight = rowHeight, 
                                        columnHeight = columnHeight, 
                                        numberOfRows = 4, 
                                        numberOfRepetitions = numberOfRepetitions)*255).astype(np.uint8)
##ax[1].imshow(calibration_template)
##ax[1].set_title("Calibration template")

synchronization_template = np.repeat(np.expand_dims((generateSyncTemplate(rowHeight = rowHeight)*255).astype(np.uint8),1),columnHeight,axis=1)
synchronization_template = np.repeat(np.expand_dims(synchronization_template, axis=2),3,axis=2)
synchronization_template[:,:,0] = 0
synchronization_template[:,:,2] = 0

##ax[2].imshow(synchronization_template)
##ax[2].set_title("Synchronization template")
##ax[0].axis('scaled')
##ax[1].axis('scaled')
##ax[2].axis('scaled')

method = cv2.TM_CCORR_NORMED
result = cv2.matchTemplate(frame,detection_template,method)

#Channel
greenChannel = frame.copy()
greenChannel[:, :, 0] = 0
greenChannel[:, :, 2] = 0

greenChannelTemplate = detection_template.copy()
greenChannelTemplate[:, :, 0] = 0
greenChannelTemplate[:, :, 2] = 0

fig, matchTemplateAx = plt.subplots(1,3, figsize=(30,15))
# matchTemplateAx[0].imshow(frame)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
top_left = max_loc
h, w, z = detection_template.shape
bottom_right = (top_left[0] + w, top_left[1] + h)
roi= np.array((top_left, bottom_right))

roiFoundFrame = frame.copy()
cv2.rectangle(roiFoundFrame, tuple(roi[0]), tuple(roi[1]), (255,255,255), 2)


roiFrame = frame[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0],:].copy()
matchTemplateAx[1].imshow(roiFrame)
matchTemplateAx[2].imshow(roiFoundFrame)

plt.show()


# powerDistribution = frame.copy()/255.0
# height, width, channels = powerDistribution.shape
# line = powerDistribution[:,600:601,:]
# powerDistribution = powerDistribution[:,325:925,:]
# greenChannel = np.zeros((height,powerDistribution.shape[1],3))
# greenChannel [:,:,1] = powerDistribution[:,:,1]
# redChannel = np.zeros((height,powerDistribution.shape[1],3))
# redChannel [:,:,0] = powerDistribution[:,:,0]
# blueChannel = np.zeros((height,powerDistribution.shape[1],3))
# blueChannel [:,:,2] = powerDistribution[:,:,2]

# fig, redAx = plt.subplots(1,2)
# redAx[0].imshow(redChannel)
# redAx[1].plot(range(height),line[:,:,0], c= 'r')
# fig, greenAx = plt.subplots(1,2)
# greenAx[0].imshow(greenChannel)
# greenAx[1].plot(range(height),line[:,:,1], c= 'g')
# fig, blueAx = plt.subplots(1,2)
# blueAx[0].imshow(blueChannel)
# blueAx[1].plot(range(height),line[:,:,2], c= 'b')

# plt.show()

calibrationMaxIterations = 10

video = cv2.VideoCapture('testV2lineasGordas.h264')
roiHeight = roi[1][1]-roi[0][1]

# ROI calibration measurements database for every channel.
redCalibrationMeasurements = np.zeros ((roiHeight,3))
greenCalibrationMeasurements = np.zeros ((roiHeight,3))
blueCalibrationMeasurements = np.zeros ((roiHeight,3))
redCalibrationMeasurements += 0.0005
greenCalibrationMeasurements += 0.0005
blueCalibrationMeasurements += 0.0005

# Template used for packet detection.
calibration_template = generateROIDetectionTemplate(rowHeight = 11, 
                                              columnHeight = 20,
                                              numberOfRows = 5, 
                                              numberOfRepetitions = 1)

""" fig, calAx = plt.subplots(calibrationMaxIterations+1,3) """

for calibrationIteration in range(0,calibrationMaxIterations+1):
    status, frame = video.read(0)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32)/255.0
    # Mask ROI
    roiFrame = frame[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0],:].copy()
    # Look for packet template
    result = cv2.matchTemplate(roiFrame,calibration_template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    topRow = max_loc[1]
    topColumn = max_loc[0]
    h, columnHeight, z = calibration_template.shape
    
    startingRow = rowHeight//2-rowHeight//2//3 + topRow
    endingRow = rowHeight//2+rowHeight//2//3 + topRow
    startingColumn = columnHeight//2 - 1
    endingColumn = columnHeight//2
    rows = endingRow - startingRow
    
    greenCalibrationMeasurements[startingRow:endingRow]=\
            (roiFrame[startingRow:endingRow,
             startingColumn:endingColumn]).reshape(rows,3)

    blueCalibrationMeasurements[startingRow+rowHeight:endingRow+rowHeight]\
            =(roiFrame[startingRow+rowHeight:endingRow+rowHeight,
              startingColumn:endingColumn]).reshape(rows,3)

    redCalibrationMeasurements[startingRow+2*rowHeight:endingRow+2*rowHeight]\
            =(roiFrame[startingRow+2*rowHeight:endingRow+2*rowHeight,
              startingColumn:endingColumn]).reshape(rows,3)
    
    greenCalibrationMeasurements[startingRow+4*rowHeight:endingRow+4*rowHeight]\
            =(roiFrame[startingRow+4*rowHeight:endingRow+4*rowHeight,
              startingColumn:endingColumn]).reshape(rows,3)
    
#     calAx[calibrationIteration,1].title.set_text(
#         'Calibration after the frame: ({})'.format(calibrationIteration+1))
    
#     calAx[calibrationIteration,0].plot(greenCalibrationMeasurements[:,0],c='r')
#     calAx[calibrationIteration,0].plot(greenCalibrationMeasurements[:,1],c='g')
#     calAx[calibrationIteration,0].plot(greenCalibrationMeasurements[:,2],c='b')
#     calAx[calibrationIteration,1].plot(redCalibrationMeasurements[:,0],c='r')
#     calAx[calibrationIteration,1].plot(redCalibrationMeasurements[:,1],c='g')
#     calAx[calibrationIteration,1].plot(redCalibrationMeasurements[:,2],c='b')
#     calAx[calibrationIteration,2].plot(blueCalibrationMeasurements[:,0],c='r')
#     calAx[calibrationIteration,2].plot(blueCalibrationMeasurements[:,1],c='g')
#     calAx[calibrationIteration,2].plot(blueCalibrationMeasurements[:,2],c='b')

# plt.subplots_adjust(top=1.5,hspace=0.7)

# plt.show()

def polinomialFitting(calibrationMeasurements, ord=1):
    
    x = np.arange(len(calibrationMeasurements))
    idx0 = np.greater(calibrationMeasurements[:,0],0.0005)
    idx1 = np.greater(calibrationMeasurements[:,1],0.0005)
    idx2 = np.greater(calibrationMeasurements[:,2],0.0005)

    greenFitting = np.polyfit(x[idx1], calibrationMeasurements[idx1,1], ord)
    blueFitting = np.polyfit(x[idx2], calibrationMeasurements[idx2,2], ord)
    redFitting = np.polyfit(x[idx0], calibrationMeasurements[idx0,0], ord)

    greenFitter = np.poly1d(greenFitting)
    blueFitter = np.poly1d(blueFitting)
    redFitter = np.poly1d(redFitting)

    greenAjustment = greenFitter(x)
    blueAjustment = blueFitter(x)
    redAjustment = redFitter(x)
    
    return np.concatenate((redAjustment.reshape(calibrationMeasurements.shape[0],1),\
                           greenAjustment.reshape(calibrationMeasurements.shape[0],1),\
                           blueAjustment.reshape(calibrationMeasurements.shape[0],1)),axis=1)

greenAjustments = polinomialFitting(greenCalibrationMeasurements)
blueAjustments = polinomialFitting(blueCalibrationMeasurements)
redAjustments = polinomialFitting(redCalibrationMeasurements)

# fig, ajustAx = plt.subplots(3,1)

# ajustAx[1].title.set_text('Red channel interferences')
# ajustAx[1].plot(redCalibrationMeasurements[:,0],c='#293336')
# ajustAx[1].plot(redCalibrationMeasurements[:,1],c='#a7adba')
# ajustAx[1].plot(redCalibrationMeasurements[:,2],c='#a7adba')

# ajustAx[1].plot(redAjustments[:,0],c='r')
# ajustAx[1].plot(redAjustments[:,1],c='g')
# ajustAx[1].plot(redAjustments[:,2],c='b')

# ajustAx[0].title.set_text('Green channel interferences')
# ajustAx[0].plot(greenCalibrationMeasurements[:,0],c='#a7adba')
# ajustAx[0].plot(greenCalibrationMeasurements[:,1],c='#293336')
# ajustAx[0].plot(greenCalibrationMeasurements[:,2],c='#a7adba')

# ajustAx[0].plot(greenAjustments[:,0],c='r')
# ajustAx[0].plot(greenAjustments[:,1],c='g')
# ajustAx[0].plot(greenAjustments[:,2],c='b')

# ajustAx[2].title.set_text('Blue channel interferences')
# ajustAx[2].plot(blueCalibrationMeasurements[:,0],c='#a7adba')
# ajustAx[2].plot(blueCalibrationMeasurements[:,1],c='#a7adba')
# ajustAx[2].plot(blueCalibrationMeasurements[:,2],c='#293336')

# ajustAx[2].plot(blueAjustments[:,0],c='r')
# ajustAx[2].plot(blueAjustments[:,1],c='g')
# ajustAx[2].plot(blueAjustments[:,2],c='b')

# plt.subplots_adjust(top=1.5,hspace=0.7)
# plt.show()

invColorMatrix = 0
spatialThresholding = 0
numberOfSamples = len(greenCalibrationMeasurements)
redNormalization = np.divide(redAjustments,redAjustments[:,0].reshape(numberOfSamples,1))
greenNormalization = np.divide(greenAjustments,greenAjustments[:,1].reshape(numberOfSamples,1))
blueNormalization = np.divide(blueAjustments,blueAjustments[:,2].reshape(numberOfSamples,1))

for i in range(0,numberOfSamples):
    currentMatrix = np.concatenate(
        (redNormalization[i, :].reshape(1,3),
        greenNormalization[i,:].reshape(1,3),
        blueNormalization[i,:].reshape(1,3)),axis=0).reshape(1,3,3)
    try:
        inverse = np.linalg.inv(currentMatrix)
    except np.linalg.LinAlgError:
        print("Inverse matrix not computable")
        pass
    else:
        if(i==0):
            invColorMatrix = inverse
        else:
            invColorMatrix = np.concatenate((invColorMatrix, inverse))

spatialThresholding = np.concatenate((redAjustments[:,0].reshape(numberOfSamples,1),
                                      greenAjustments[:,1].reshape(numberOfSamples,1),
                                      blueAjustments[:,2].reshape(numberOfSamples,1)),axis=1)

for i in spatialThresholding[:,1]:
    spatialThresholding[:,1]=0.5
for i in spatialThresholding[:,0]:
    spatialThresholding[:,0]=0.5
for i in spatialThresholding[:,2]:
    spatialThresholding[:,2]=0.5

plt.matshow(currentMatrix[0], cmap='gnuplot')
plt.ylabel("Transmitted LED color")
plt.xlabel("Received channel")
plt.yticks(ticks=np.arange(3),labels=["Red","Green","Blue"])
plt.xticks(ticks=np.arange(3),labels=["Red","Green","Blue"])
plt.colorbar()
plt.title("Channel interference matrix for the first pixel")


from matplotlib.colors import LinearSegmentedColormap
video.set(cv2.CAP_PROP_POS_FRAMES, 175)
status, frame = video.read(40)
frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
# frame = cv2.GaussianBlur(frame,(5,5),0)


# Previous ROI detected
roiHeight = roi[1][1]-roi[0][1]
frame = frame.astype(np.float32)/255.0

# Mask ROI
roiFrame = frame[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0],:].copy()
numberOfSamples,width,channels = roiFrame.shape
roiFrame = roiFrame[:,width//2,:]
colorAdaptedSignal = 0

for i in range (0,numberOfSamples):
    colorAdaptedSample = np.dot(roiFrame[i],invColorMatrix[i]).reshape(1, 3)
    if i == 0:
        colorAdaptedSignal = colorAdaptedSample
    else:
        colorAdaptedSignal = np.concatenate((colorAdaptedSignal,colorAdaptedSample))

colorAdaptedSignalFiltered = cv2.blur(roiFrame,(1,3))
colorAdaptedSignalFiltered[colorAdaptedSignalFiltered<=0] = 0

# Plots
fig, colorAdaptedAx = plt.subplots(3,2)
colorAdaptedAx[0,0].title.set_text('Raw signal')
colorAdaptedAx[0,0].plot(roiFrame[:,0], c='r')
colorAdaptedAx[0,1].title.set_text('Equalized signal')
colorAdaptedAx[0,1].plot(colorAdaptedSignal[:,0], c='gray',label="equalized")
colorAdaptedAx[0,1].plot(colorAdaptedSignalFiltered[:,0], c='r',label="filtered")
colorAdaptedAx[0,1].legend(loc='upper right')
colorAdaptedAx[1,0].plot(roiFrame[:,1], c='g')
colorAdaptedAx[1,1].plot(colorAdaptedSignal[:,1], c='gray',label="equalized")
colorAdaptedAx[1,1].plot(colorAdaptedSignalFiltered[:,1], c='g',label="filtered")
colorAdaptedAx[1,1].legend(loc='upper right')
colorAdaptedAx[2,0].plot(roiFrame[:,2], c='b')
colorAdaptedAx[2,1].plot(colorAdaptedSignal[:,2], c='gray',label="equalized")
colorAdaptedAx[2,1].plot(colorAdaptedSignalFiltered[:,2], c='b',label="filtered")
colorAdaptedAx[2,1].legend(loc='upper right')

roiFrame = np.repeat(roiFrame, 30, axis=1)
colorAdaptedSignalPlot = colorAdaptedSignal.copy()
colorAdaptedSignalPlot = np.repeat(colorAdaptedSignalPlot, 30, axis=1)

colorAdaptedSignalFilteredPlot = colorAdaptedSignalFiltered.copy()
colorAdaptedSignalFilteredPlot = np.repeat(colorAdaptedSignalFilteredPlot, 30, axis=1)

# fig, colorAdaptedPlotAx = plt.subplots(1,3,figsize=(7,7),sharey=True)
# cm = LinearSegmentedColormap.from_list("Custom", [(0.2, 0, 0), (1, 0, 0)], N=100)
# colorAdaptedPlotAx[0].title.set_text('Raw signal \n(Red channel)')
# colorAdaptedPlotAx[0].imshow(roiFrame[:,:30],cmap=cm)
# colorAdaptedPlotAx[1].title.set_text('Equalized \n(Red channel)')
# colorAdaptedPlotAx[1].imshow(colorAdaptedSignalPlot[:,:30],cmap=cm)
# colorAdaptedPlotAx[2].title.set_text('Filtered \n(Red channel)')
# colorAdaptedPlotAx[2].imshow(colorAdaptedSignalFilteredPlot[:,:30],cmap=cm)

# fig, colorAdaptedPlotAx = plt.subplots(1,3,figsize=(7,7),sharey=True)
# cm = LinearSegmentedColormap.from_list("Custom", [(0, 0.2, 0), (0, 1, 0)], N=100)
# colorAdaptedPlotAx[0].title.set_text('Raw signal \n(Green channel)')
# colorAdaptedPlotAx[0].imshow(roiFrame[:,30:60],cmap=cm)
# colorAdaptedPlotAx[1].title.set_text('Equalized \n(Green channel)')
# colorAdaptedPlotAx[1].imshow(colorAdaptedSignalPlot[:,30:60],cmap=cm)
# colorAdaptedPlotAx[2].title.set_text('Filtered \n(Green channel)')
# colorAdaptedPlotAx[2].imshow(colorAdaptedSignalFilteredPlot[:,30:60],cmap=cm)

# fig, colorAdaptedPlotAx = plt.subplots(1,3,figsize=(7,7),sharey=True)
# cm = LinearSegmentedColormap.from_list("Custom", [(0, 0, 0.2), (0, 0, 1)], N=100)
# colorAdaptedPlotAx[0].title.set_text('Raw signal \n(Blue channel)')
# colorAdaptedPlotAx[0].imshow(roiFrame[:,60:90],cmap=cm)
# colorAdaptedPlotAx[1].title.set_text('Equalized \n(Blue channel)')
# colorAdaptedPlotAx[1].imshow(colorAdaptedSignalPlot[:,60:90],cmap=cm)
# colorAdaptedPlotAx[2].title.set_text('Filtered \n(Blue channel)')
# colorAdaptedPlotAx[2].imshow(colorAdaptedSignalFilteredPlot[:,60:90],cmap=cm)

# plt.show()

############
binarizeSignal = 0
signal = colorAdaptedSignalFiltered.copy()
numberOfSamples = len(signal)

redChannel = np.zeros(numberOfSamples).reshape(numberOfSamples,1)
greenChannel = np.zeros(numberOfSamples).reshape(numberOfSamples,1)
blueChannel = np.zeros(numberOfSamples).reshape(numberOfSamples,1)

fig, histAx = plt.subplots(1,3)
histAx[0].title.set_text('Red channel histogram')
histAx[0].hist(roiFrame[:,0], bins=10, color='r')
histAx[1].title.set_text('Green channel histogram')
histAx[1].hist(roiFrame[:,1], bins=10, color='g')
histAx[2].title.set_text('Blue channel histogram')
histAx[2].hist(roiFrame[:,2], bins=10, color='b')

redChannel[np.less(spatialThresholding[:,0],roiFrame[:,0], where=True)] = 1.0
greenChannel[np.less(spatialThresholding[:,1],roiFrame[:,1], where=True)] = 1.0
blueChannel[np.less(spatialThresholding[:,2],roiFrame[:,2], where=True)] = 1.0

binarizeSignal = np.concatenate((redChannel,greenChannel,blueChannel),axis=1)

binarizedSignalPlot = binarizeSignal.copy()
binarizedSignalPlot = np.expand_dims(binarizedSignalPlot, axis=1)
binarizedSignalPlot = np.repeat(binarizedSignalPlot, 300, axis=1)

fig, colorAdaptedPlotAx = plt.subplots(3,3)
colorAdaptedPlotAx[0,0].title.set_text('Raw signal (Red channel)')
colorAdaptedPlotAx[0,0].plot(roiFrame[:,0],c='r',label='Equalized')
colorAdaptedPlotAx[0,0].plot(spatialThresholding[:,0],c='orange',label='Threshold')
colorAdaptedPlotAx[0,0].legend(loc='upper right')
colorAdaptedPlotAx[0,1].title.set_text('Binarized signal')
colorAdaptedPlotAx[0,1].plot(redChannel,c='r')
colorAdaptedPlotAx[0,2].title.set_text('Binarized signal (2D)')
colorAdaptedPlotAx[0,2].imshow(binarizedSignalPlot[:,:,0], cmap='Reds')
colorAdaptedPlotAx[1,0].title.set_text('Raw signal (Green channel)')
colorAdaptedPlotAx[1,0].plot(roiFrame[:,1],c='g',label='Equalized')
colorAdaptedPlotAx[1,0].plot(spatialThresholding[:,1],c='orange',label='Threshold')
colorAdaptedPlotAx[1,0].legend(loc='upper right')
colorAdaptedPlotAx[1,1].title.set_text('Binarized signal')
colorAdaptedPlotAx[1,1].plot(greenChannel,c='g')
colorAdaptedPlotAx[1,2].title.set_text('Binarized signal (2D)')
colorAdaptedPlotAx[1,2].imshow(binarizedSignalPlot[:,:,1], cmap='Greens')
colorAdaptedPlotAx[2,0].title.set_text('Raw signal (Blue channel)')
colorAdaptedPlotAx[2,0].plot(roiFrame[:,2],c='b',label='Equalized')
colorAdaptedPlotAx[2,0].plot(spatialThresholding[:,2],c='orange',label='Threshold')
colorAdaptedPlotAx[2,0].legend(loc='upper right')
colorAdaptedPlotAx[2,1].title.set_text('Binarized signal')
colorAdaptedPlotAx[2,1].plot(blueChannel,c='b')
colorAdaptedPlotAx[2,2].title.set_text('Binarized signal (2D)')
colorAdaptedPlotAx[2,2].imshow(binarizedSignalPlot[:,:,2], cmap='Blues')

plt.subplots_adjust(top=1,hspace=0.5)
plt.show()
synchronization_template = generateSyncTemplate(rowHeight = 11)

synchronization_templateN = (synchronization_template - np.mean(synchronization_template)) / (np.std(synchronization_template) * len(synchronization_template))
greenChannelN = (greenChannel - np.mean(greenChannel)) /  np.std(greenChannel)
greenChannelN = np.reshape(greenChannelN,len(greenChannelN))
correlation = np.correlate(greenChannelN,synchronization_templateN)

maxValue = np.amax(correlation)
maxIndexValue = np.argmax(correlation)

correlationPlot = np.zeros(len(greenChannelN))
correlationPlot[maxIndexValue:maxIndexValue+len(synchronization_template)] = synchronization_template
plt.plot(redChannel-1.5,c='r',label="Red channel (Data)")
plt.plot(greenChannel,c='g',label="Green channel (Synchronization pattern)")
plt.plot(blueChannel-3,c='b',label="Blue channel (Data)")
plt.plot(correlationPlot,c='black',linestyle='-.',label="Synchronization template")

sampling_points = np.arange(rowHeight//2, len(synchronization_template), rowHeight)+maxIndexValue
for idx,sampling_point in enumerate(sampling_points):
  plt.axvline(sampling_point,color='orange',linestyle='--',label="Sampling points" if idx==0 else "")
plt.legend(loc='center left')
plt.title("Synchronization on Green channel and decoding on Red and Blue channels")
plt.show()

print(greenChannel)
binariCode = "" 

for i in range(0,len(greenChannel),4):

    if redChannel[i] == 0.:
        binariCode = binariCode + " "+ str(0)
    else:
        binariCode = binariCode + " "+ str(1)

print("Este es el codigo binario")
print("[" + binariCode+"]") 