# YOLO object detection using a webcam
# Exact same demo as the read from disk, but instead of disk a webcam is used.
# import the necessary packages
import numpy as np
# import argparse
import imutils
import time
import cv2
import os
import PySimpleGUI as sg

i_vid = r'videos\input1.mp4'
o_vid = r'output\input1_out.mp4'
y_path = r'yolo-coco'
sg.ChangeLookAndFeel('LightGreen')
# insialisasi untuk GUI input video,path penyimpanan model,nilai confidence dan threshold, dan juga pilihan untuk menggunakan webcam atau tidak
layout = 	[
		[sg.Text('YOLO Video Player', size=(18,1), font=('Any',18),text_color='#1c86ee' ,justification='left')],
		[sg.Text('Path to input video'), sg.In(i_vid,size=(40,1), key='input'), sg.FileBrowse()],
		[sg.Text('Optional Path to output video'), sg.In(o_vid,size=(40,1), key='output'), sg.FileSaveAs()],
		[sg.Text('Yolo base path'), sg.In(y_path,size=(40,1), key='yolo'), sg.FolderBrowse()],
		[sg.Text('Confidence'), sg.Slider(range=(0,1),orientation='h', resolution=.1, default_value=.5, size=(15,15), key='confidence')],
		[sg.Text('Threshold'), sg.Slider(range=(0,1), orientation='h', resolution=.1, default_value=.3, size=(15,15), key='threshold')],
		[sg.Text(' '*8), sg.Checkbox('Use webcam', key='_WEBCAM_')],
		[sg.Text(' '*8), sg.Checkbox('Write to disk', key='_DISK_')],
		[sg.Ok('Run'), sg.Cancel('Exit')]
			]

win = sg.Window('Traffic Counting GUI',
				default_element_size=(21,1),
				text_justification='right',
				auto_size_text=False).Layout(layout)
event, values = win.Read()
if event is None or event =='Cancel':
	exit()
write_to_disk = values['_DISK_']
use_webcam = values['_WEBCAM_']
args = values
win.Close()
# inisialisasi variable counting
inccount1 = 0
inccount2 = 0
inccount3 = 0
inccount4 = 0
inccount5 = 0
inccount6 = 0
inccount7 = 0
inccount8 = 0
inccount9 = 0
inccount10 = 0
inccount11 = 0
inccount12 = 0
inccount13 = 0
inccount14 = 0
inccount15 = 0
inccount16 = 0
inccount_reset = 0
start_time = time.time()
# imgbytes = cv2.imencode('.png', image)[1].tobytes()  # ditto
gui_confidence = args["confidence"]
gui_threshold = args["threshold"]
# load label obj.names dari model yolov4
labelsPath = os.path.sep.join([args["yolo"], "obj.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# inisialisasi list warna untuk setiap label class
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# path untuk mengambil model yolov4
weightsPath = os.path.sep.join([args["yolo"], "yolov4_last.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov4.cfg"])

# load yolov4 object detector dan ambil hanya bagian outputnya saja
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# inisialisasi video stream dari input GUI
vs = cv2.VideoCapture(args["input"])
writer = None

# mengkalkulasi jumlah frame pada video
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# membuat fungsi untuk melakukan object detection
def findObject(layerOutputs,frame):
	(W, H) = (None, None)
	if W is None or H is None:
		(H, W) = frame.shape[:2]



	# inisialisasi list boxes,confidences dan classIDs
	# inisialisasi variable untuk counting
	boxes = []
	confidences = []
	classIDs = []
	count1 = 0
	count2 = 0
	count3 = 0
	count4 = 0
	count5 = 0
	count6 = 0
	count7 = 0
	count8 = 0
	count9 = 0
	count10 = 0
	count11 = 0
	count12 = 0
	count13 = 0
	count14 = 0

	# looping pada variable outputs
	for output in layerOutputs:
		# looping pada setiap deteksi
		for det in output:
			# extract class id dan nilai confidence dari deteksi
			scores = det[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# membandingkan nilai confidence dari deteksi dan batas confidence yang ditetapkan
			if confidence > gui_confidence:
				# mencari koordinat dari bbox
				w, h = int(det[2] * W), int(det[3] * H)
				x, y = int((det[0] * W) - w / 2), int((det[1] * H) - h / 2)
				boxes.append([x, y, w, h])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# menambahkan nmsboxes untuk mencegah bbox yang saling menimpa
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, gui_confidence, gui_threshold)

	# memastikan setidaknya ada 1 bbox yang terdeteksi
	if len(idxs) > 0:
		for i in idxs.flatten():
			# extract koordinat bbox
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# menyiapkan titik tengah x,y dari bounding box
			xMid = int((x + (x + w)) / 2)
			yMid = int((y + (y + h)) / 2)
			# menampilkan titik tengah dari bounding box
			cv2.circle(frame, (xMid, yMid), 1, (0, 0, 255), 5)
			# counting vehicle berdasarkan titik tengah bb yang bersinggungan dengan green line
			# untuk counting sebelah kiri
			if yMid > 0 and yMid < 0 and xMid > 0 and xMid < 0:
				if classID == 0:
					count1 = count1 + 1
				elif classID == 1:
					count2 = count2 + 1
				elif classID == 2:
					count3 = count3 + 1
				elif classID == 3:
					count4 = count4 + 1
				elif classID == 4:
					count5 = count5 + 1
				elif classID == 5:
					count6 = count6 + 1
				elif classID == 6:
					count7 = count7 + 1
			# untuk counting sebelah kanan
			if yMid > 511 and yMid < 515 and xMid > 1 and xMid < 1279:
				cv2.line(frame, (1, 512), (1279, 512), (0, 0, 255), 3)  # Red line
				if classIDs[i] == 0:
					count8 = count8 + 1
				elif classIDs[i] == 1:
					count9 = count9 + 1
				elif classIDs[i] == 2:
					count10 = count10 + 1
				elif classIDs[i] == 3:
					count11 = count11 + 1
				elif classIDs[i] == 4:
					count12 = count12 + 1
				elif classIDs[i] == 5:
					count13 = count13 + 1
				elif classIDs[i] == 6:
					count14 = count14 + 1

			# gambar bbox beserta label dan nilai confidence nya pada frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			# membuat garis green line sebagai area counting
			cv2.line(frame, (1, 512), (1279, 512), (0, 255, 0),3)  # Green Offset Line
	return count1, count2, count3, count4, count5, count6, count7, count8,count9, count10, count11, count12, count13, count14
win_started = False
if use_webcam:
	cap = cv2.VideoCapture(0)
while True:
	if use_webcam:
		grabbed, frame = cap.read()
	else:
		grabbed, frame = vs.read()

	if not grabbed:
		print(f'Total Kendaraan : {inccount16}')
		break

	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
								 swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	#inisialisasi nilai counter
	counter1, counter2, counter3, counter4, counter5, counter6, counter7, counter8, counter9, counter10, counter11, counter12, counter13, counter14 = findObject(layerOutputs,frame)
	# proses penempatan hasil counter untuk ditampilkan
	inccount1 = inccount1 + counter1
	inccount2 = inccount2 + counter2
	inccount3 = inccount3 + counter3
	inccount4 = inccount4 + counter4
	inccount5 = inccount5 + counter5
	inccount6 = inccount6 + counter6
	inccount7 = inccount7 + counter7
	inccount8 = inccount8 + (counter1 + counter2 + counter3 + counter4+counter5 + counter6 + counter7)
	inccount9 = inccount9 + counter8
	inccount10 = inccount10 + counter9
	inccount11 = inccount11 + counter10
	inccount12 = inccount12 + counter11
	inccount13 = inccount13 + counter12
	inccount14 = inccount14 + counter13
	inccount15 = inccount15 + counter14
	inccount16 = inccount16 + (counter8 + counter9 + counter10 + counter11+counter12 + counter13 + counter14)
	run_time = time.time()
	iccount_reset = int(time.time() - start_time)
	if inccount_reset == 3600:
		inccount1 = 0
		inccount2 = 0
		inccount3 = 0
		inccount4 = 0
		inccount5 = 0
		inccount6 = 0
		inccount7 = 0
		inccount8 = 0
		inccount9 = 0
		inccount10 = 0
		inccount11 = 0
		inccount12 = 0
		inccount13 = 0
		inccount14 = 0
		inccount15 = 0
		inccount16 = 0
		inccount_reset = 0
		start_time = run_time
	# menampilkan hasil counting pada frame
	cv2.putText(frame, f'counting mobil : {inccount1}', (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
	cv2.putText(frame, f'counting truk 3 sumbu : {inccount2}', (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
	cv2.putText(frame, f'counting truk 2 sumbu : {inccount3}', (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
	cv2.putText(frame, f'counting truk 4 sumbu : {inccount4}', (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
	cv2.putText(frame, f'counting bis kecil : {inccount5}', (15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
	cv2.putText(frame, f'counting bis besar : {inccount6}', (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
	cv2.putText(frame, f'counting sepeda motor : {inccount7}', (15, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
	cv2.putText(frame, f'Total : {inccount8}', (15, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
	cv2.putText(frame, f'counting mobil : {inccount9}', (980, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
	cv2.putText(frame, f'counting truk 3 sumbu : {inccount10}', (980, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
	cv2.putText(frame, f'counting truk 2 sumbu : {inccount11}', (980, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
	cv2.putText(frame, f'counting truk 4 sumbu : {inccount12}', (980, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
	cv2.putText(frame, f'counting bis kecil : {inccount13}', (980, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
	cv2.putText(frame, f'counting bis besar : {inccount14}', (980, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
	cv2.putText(frame, f'counting sepeda motor : {inccount15}', (980, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
	cv2.putText(frame, f'Total : {inccount16}', (980, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
	if write_to_disk:
		if writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(frame.shape[1], frame.shape[0]), True)

			if total > 0:
				elap = (end - start)
				print("[INFO] single frame took {:.4f} seconds".format(elap))
				print("[INFO] estimated total time to finish: {:.4f}".format(
					elap * total))
		writer.write(frame)
	imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
	# inisialisasi GUI untuk menampilkan hasil deteksi
	if not win_started:
		win_started = True
		layout = [
			[sg.Text('SISTEM PERHITUNGAN JUMLAH KENDARAAN BERBASIS YOLOV4 \n DEEP NEURAL NETWORKS', size=(160,3),justification='center')],
			[sg.Image(data=imgbytes, key='_IMAGE_')],
			[sg.Text("JUMLAH TOTAL KENDARAAN", size=(30, 1),justification='left')],
			[sg.Text('Confidence'),
			 sg.Slider(range=(0, 1), orientation='h', resolution=.1, default_value=.5, size=(15, 15), key='confidence'),
			sg.Text('Threshold'),
			 sg.Slider(range=(0, 1), orientation='h', resolution=.1, default_value=.3, size=(15, 15), key='threshold')],
			[sg.Exit()]
		]
		win = sg.Window('YOLO Output',
						default_element_size=(14, 1),
						text_justification='right',
						auto_size_text=False).Layout(layout).Finalize()
		image_elem = win.FindElement('_IMAGE_')
	else:
		image_elem.Update(data=imgbytes)

	event, values = win.Read(timeout=0)
	if event is None or event == 'Exit':
		print(f'Total Kendaraan : {inccount16}')
		break
	gui_confidence = values['confidence']
	gui_threshold = values['threshold']


win.Close()

# release the file pointers
print("[INFO] cleaning up...")
writer.release() if writer is not None else None
vs.release()