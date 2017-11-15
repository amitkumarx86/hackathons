from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

@csrf_exempt
def index(request):
	if request.method == 'GET':
		return render(request,'html/main.html')
	else:
		try:
			import base64
			from PIL import Image
			from io import BytesIO
			data = request.POST['link'].replace('data:image/jpeg;base64,','')
			im = Image.open(BytesIO(base64.b64decode(data))).convert('RGB')
			im.save(open("img/cheque.png",'wb+'))
			im.save(open("cheque/static/cheque.png",'wb+'))
			jsonData = extract("img/cheque.png")
			print(jsonData)
			return HttpResponse(jsonData)
		except:
			# get file using upload method
			# try:
			myfile = request.FILES['file']
			destination = open("img/cheque.png", 'wb+')
			for chunk in myfile.chunks():
				destination.write(chunk)
			destination.close()
			destination = open("cheque/static/cheque.png", 'wb+')
			for chunk in myfile.chunks():
				destination.write(chunk)
			destination.close()
			jsonData = extract("img/cheque.png")
			print(jsonData)
			return HttpResponse(jsonData)
			# except:
				# return HttpResponse("nothing happend")

def extract(imgName):
	# imports
	import cv2
	from imutils import contours
	import cv2
	from sklearn.externals import joblib
	from skimage.feature import hog
	import numpy as np
	from imutils import contours
	import matplotlib.pyplot as plt
	import numpy as np
	import cv2
	import sys
	from matplotlib import pyplot as plt



	# utility functions
	def text2int(textnum, numwords={}):
		if not numwords:
			units = [
			"zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
			"nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
			"sixteen", "seventeen", "eighteen", "nineteen",
			]

			tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

			scales = ["hundred", "thousand", "million", "billion", "trillion"]

			numwords["and"] = (1, 0)
			for idx, word in enumerate(units):    numwords[word] = (1, idx)
			for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
			for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)

		current = result = 0
		for word in textnum.split():
			if word not in numwords:
			    raise Exception("Illegal word: " + word)

			scale, increment = numwords[word]
			current = current * scale + increment
			if scale > 100:
			    result += current
			    current = 0

		return result + current

	def verifySign(path1, path2):

	    im1 = cv2.imread(path1, 0)
	    im2 = cv2.imread(path2, 0)

	   
	    # resize the images
	    img1 = cv2.resize(im1, (1300, 500), interpolation = cv2.INTER_AREA)
	    img2 = cv2.resize(im2, (1300, 500), interpolation = cv2.INTER_AREA)
	    
	    
	    
	    # Initiate SIFT detector
	    orb = cv2.ORB_create()

	    # find the keypoints with ORB
	    kp1 = orb.detect(img1,None)
	    # compute the descriptors with ORB
	    kp1, des1 = orb.compute(img1, kp1)
	    
	    kp2 = orb.detect(img2,None)
	    # compute the descriptors with ORB
	    kp2, des2 = orb.compute(img2, kp2)
	    
	    # create BFMatcher object
	    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	    # Match descriptors.
	    matches= bf.match(des1,des2)

	    # Sort them in the order of their distance.
	    matches = sorted(matches, key = lambda x:x.distance)
	    match_percent = (len(matches)*100)/len(des1)
	    return match_percent
	# -------------------------------------------------------------------------------------

	def getAmountDigit(img,name):
	    number = 0
	    # print(img)
	    im_gray = cv2.GaussianBlur(img, (5, 5), 0)
	    # Threshold the image
	    ret, im_th = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY_INV)
	    im2, ctrs, hierarchy = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	    clf = joblib.load("ml_model/digits_cls.pkl")
	    
	    temp = cv2.imread(name)
	    cv2.drawContours(temp,ctrs,-1,(255,0,0),3)
	    ctrs = contours.sort_contours(ctrs,method="left-to-right")[0]

	    # for non digits filter the contours
	    ctrs = [x for x in ctrs if cv2.contourArea(x) > 200]
	    
	    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
	    
	    number = 0
	    for rect in rects[0:len(rects)]:
	        # Draw the rectangles
	        cv2.rectangle(temp, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
	        # Resize the image
	        roi = im_th[rect[1]:rect[1] + rect[3], rect[0]-10:5+rect[0] + rect[2]]
	        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
	        roi = cv2.dilate(roi, (3, 3))

	        # Calculate the HOG features    
	        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
	        nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
	        number = int(nbr[0]) + number*10
	        
	    return number

	# -------------------------------------------------------------------------------------
	def getDigit(img):

	    number = 0
	    clf = joblib.load("ml_model/digits_cls.pkl")
	    im_gray = cv2.GaussianBlur(img, (5, 5), 0)

	    # Threshold the image
	    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
	    im2, ctrs, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	    ctrs = contours.sort_contours(ctrs,method="left-to-right")[0]
	    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
	    number = 0

	    for rect in rects[0:len(rects)]:
	        # Draw the rectangles
	        cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 255, 0), 3)
	        # Resize the image
	        roi = im_th[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
	        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
	        roi = cv2.dilate(roi, (3, 3))
	        # Calculate the HOG features    
	        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
	        nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
	        number = int(nbr[0]) + number*10
	        
	    return number
	# -------------------------------------------------------------------------------------
	def show(img):
	    import matplotlib.pyplot as plt
	    plt.imshow(img, cmap='gray')
	    plt.xticks([]), plt.yticks([])
	    plt.show()
	# -------------------------------------------------------------------------------------
	def getText(name):
	    import pytesseract
	    from PIL import Image, ImageEnhance, ImageFilter
	    text = pytesseract.image_to_string(Image.open(name))
	    return text
	# -------------------------------------------------------------------------------------
	# processing of image
	# imgName = "cheque4.jpg"
	im = cv2.imread(imgName)
	im = cv2.resize(im, (2708, 1216), interpolation = cv2.INTER_AREA)
	imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	cv2.imwrite("temp/cheque.jpg",imgray)

	# ---------------------------------------------------------------------
	# 1. extract features from the image
	# ---------------------------------------------------------------------
	branch     = imgray[0:180, 900:2090]
	cv2.imwrite("temp/branch.jpg",branch)

	branchName = "Not Found"
	try:
	    branchName = getText("temp/branch.jpg").replace('\n','').replace('\'','')
	except:
	    print("branch not found")

	# ---------------------------------------------------------------------
	# 2. extract date from the image
	# ---------------------------------------------------------------------
	dateVar = imgray[80:140, 2100:2650]
	dateVar = cv2.medianBlur(dateVar,5)
	kernel = np.ones((5,5),np.uint8)
	dateVar = cv2.erode(dateVar,kernel,iterations = 1)
	cv2.imwrite("date.jpg",dateVar)
	dateVar = getText("date.jpg")

	print(dateVar)

	temp = ""
	for a in dateVar.split("|"):
		if a == "o" or a == "O":
			temp = temp+""+"0"
		else:
			temp = temp+a
	import re

	dateVar = temp
	print(dateVar)

	

	dateFlag = False;
	# dateVar = "Not Found"
	try:
	    temp = dateVar.split("/")
	    dateVar = ''.join(re.findall(r'\d+', dateVar))
	    dateVar = dateVar[0]+""+dateVar[1]+"/"+dateVar[2]+""+dateVar[3]+"/"+dateVar[4]+""+dateVar[5]+""+dateVar[6]+""+dateVar[7]
	
	    day = temp[0]
	    month = temp[1]
	    year = temp[2]

	    from datetime import date
	    from dateutil.relativedelta import relativedelta
	    if date.today() > date(int(year),int(month),int(day))+relativedelta(months=+3):
	    	dateFlag = True

	except:
	    dateVar = "Not Found"
	    

	# ---------------------------------------------------------------------
	# 3. extract payee
	# ---------------------------------------------------------------------
	payee = imgray[170:330, 200:2000]
	# cv2.imwrite("./cheque/static/payee.jpg",payee)
	cv2.imwrite("temp/payee.jpg",payee)

	payeeImg = getText("temp/payee.jpg")


	# ---------------------------------------------------------------------
	# 4. type of cheque
	# ---------------------------------------------------------------------
	typeOfCheq = imgray[220:320, 2390:2670]
	cv2.imwrite("temp/typeOfCheq.jpg",typeOfCheq)
	chequeType = getText("temp/typeOfCheq.jpg").split(" ")
	chequeType = chequeType[len(chequeType)-1]


	# ---------------------------------------------------------------------
	# 5. amount in digits
	# ---------------------------------------------------------------------
	amntDigit  = imgray[445:525,2150:2590]
	kernel = np.ones((5,5),np.uint8)
	amntDigit = cv2.erode(amntDigit,kernel,iterations = 1)
	amntDigit = cv2.morphologyEx(amntDigit, cv2.MORPH_CLOSE, kernel)
	cv2.imwrite("temp/amntDigit.jpg",amntDigit)
	amntDigit = cv2.imread("temp/amntDigit.jpg",0)
	# amntDigit = "not found"

	try:
		amntDigit = getAmountDigit(amntDigit,"temp/amntDigit.jpg")
	except:
	    amntDigit = "not found"


	# ---------------------------------------------------------------------
	# 6. Account number
	# ---------------------------------------------------------------------
	account    = imgray[565:700, 210:1090]
	# show(account)
	cv2.imwrite("temp/account.jpg",account)

	account = "not found"
	try:
	    account = getText("temp/account.jpg")
	    if(len(account.split(" ")) > 1):
	    	account = account.split(" ")
	    	account = account[len(account)-1]
	    	print(account)
	    	account = [int(s) for s in account.split() if s.isdigit()]
	    	account = account[0]
	except:
	    account = "not found"

	# print(account)
	# ---------------------------------------------------------------------
	# 7. Signature
	# ---------------------------------------------------------------------
	signature  = imgray[600:900, 1850:2700]
	cv2.imwrite("temp/signature.jpg",signature)
	# verify the signature with respect to original one
	sigFlag = False
	imName = "img/"+str(account)+"-Sign.png"
	try:
	    if verifySign(imName,"temp/signature.jpg") >= 47:
	        sigFlag = True
	except:
	    print("signature verification went wrong")
	    

	# ---------------------------------------------------------------------
	# 8. ChequeInfo
	# ---------------------------------------------------------------------
	chequeInfo = imgray[1090:1170,715:985]
	kernel = np.ones((5,5),np.uint8)
	chequeInfo = cv2.erode(chequeInfo,kernel,iterations = 1)
	chequeInfo = cv2.morphologyEx(chequeInfo, cv2.MORPH_CLOSE, kernel)
	cv2.imwrite("./cheque/static/chequeInfo.jpg",chequeInfo)
	chequeInfo = "temp/chequeInfo.jpg"

	# ---------------------------------------------------------------------
	# 9. amount in words
	# ---------------------------------------------------------------------

	amntWords1  = im[325:435,400:2630]
	amntWords2  = im[440:535,100:1870]
	cv2.imwrite("amntWords1.jpg",amntWords1)
	cv2.imwrite("amntWords2.jpg",amntWords2)
	import pytesseract
	from PIL import Image, ImageEnhance, ImageFilter
	text1 = pytesseract.image_to_string(Image.open('amntWords1.jpg')).strip()
	text2 = pytesseract.image_to_string(Image.open('amntWords2.jpg')).strip()
	amntWords = (text1 + " " + text2).strip()
	
	from string import digits
	s = amntWords
	remove_digits = str.maketrans('', '', digits)
	amntWords = s.translate(remove_digits)
	amntFlag = False
	amntWords = amntWords.replace("-","").replace("_","").strip()

	amntFlag = False
	try:
		temp = ""
		for word in amntWords.split(" "):
			if len(word) >= 2:
				temp = temp + word
		amntFlag = int(text2int(str(temp).lower())) == int(amntDigit)
	except:
		amntFlag = False
	#---------------------------------------------------------------
	# import base64
	# amntWords1 = ""
	# with open(amntWords, "rb") as image_file:
	#     amntWords1 = base64.b64encode(image_file.read())
	# print(amntWords1)
	
	import json 
	chequeData = {"payeeImg":payeeImg, "amntFlag":amntFlag, "dateFlag":dateFlag,"date":dateVar,"chequeType":chequeType,"amntDigit":amntDigit,"account":account,"signature":sigFlag,"chequeInfo":chequeInfo,"amntWords":amntWords}
	json_data = json.dumps(chequeData)

	return json_data
