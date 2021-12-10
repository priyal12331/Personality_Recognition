import sklearn
import streamlit as st
import numpy as np
import cv2
import math
from PIL import Image 
#import scikit-learn as sklearn

st.title("Machine Learning based Automated Personality Assessment using Individual Handwriting Samples")




col1, col2, col3 = st.columns([1,2,1])

with col1:
    st.write("")

with col2:
    st.image("background.jpg",use_column_width=True)

with col3:
    st.write("")

features=[]
# path2 = 'images/t.jpg'

def sum_pixels(thresh):
    h, w = thresh.shape[:]
    total_intensity = 0
    pixels = 0
    for x in range(h):
        for y in range(w):
            if(thresh[x][y] >= 0):
              total_intensity += thresh[x][y]
              pixels += 1
    return total_intensity/pixels

def horizontalProjection(img):
    # Return a list containing the sum of the pixels in each row
    (h, w) = img.shape[:2]
    sumRows = []
    for j in range(h):
        row = img[j:j+1, 0:w] # y1:y2, x1:x2
        sumRows.append(np.sum(row))
    return sumRows

def verticalProjection(img):
    # Return a list containing the sum of the pixels in each column
    (h, w) = img.shape[:2]
    sumCols = []
    for j in range(w):
        col = img[0:h, j:j+1] # y1:y2, x1:x2
        sumCols.append(np.sum(col))
    return sumCols

# ref #https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
def bilateralFilter(image, d, sig):
    image = cv2.bilateralFilter(image, d, sig, sig)
    return image

def threshold(image, t):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, t, 255, cv2.THRESH_BINARY_INV)
#     ret, image = cv2.threshold(image, t, 255, cv2.THRESH_BINARY)
    return image

def dilate(image, kernalSize):
    kernel = np.ones(kernalSize, np.uint8)
    kernel
    image = cv2.dilate(image, kernel, iterations=1)
    return image

##################################################

def find_pressure(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = img.shape[:]
    inverted = img
    for x in range(h):
        for y in range(w):
            inverted[x][y] = 255 - img[x][y]

    filtered = cv2.bilateralFilter(inverted, 5, 50, 50)

    # Threshold the image
    ret,img = cv2.threshold(filtered, 127, 255, 0)

    # Step 1: Create an empty skeleton
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    original = sum_pixels(img)

    # Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    # Repeat steps 2-4
    while True:
        #Step 2: Open the image
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        #Step 3: Substract open from the original image
        temp = cv2.subtract(img, open)
        #Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(img)==0:
            break
    thinned = sum_pixels(skel)
    fin_pressure = original - thinned
    return fin_pressure

##############################################################
words = []
def find_baseline(image):
    BASELINE_ANGLE = 0.0
    contour_count=0.0
    angle = 0.0
    angle_sum = 0.0
    filtered = bilateralFilter(image, 5, 50)
    thresh = threshold(filtered, 127)
    
    dilated = dilate(thresh, (5, 50))
    
    ctrs, heir = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tmp_img = image.copy() # for updating
    tmp_img = tmp_img * 0
    tmp_img = tmp_img + 255
    list = []
    for i, ctr in enumerate(ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        # We extract the region of interest/contour to be straightened.
        roi = image[y:y+h, x:x+w]

        rect = cv2.minAreaRect(ctr)  # it returns  ( center (x,y), (width, height), angle of rotation ). 
        angle = rect[2]
        
        if angle > 80:
            angle = 90 - angle
        
        if angle < -45.0:
            angle += 90.0
        
        angle_sum += angle
        contour_count += 1

        if contour_count==0:
            contour_count = 0.000001
    mean_angle = angle_sum / contour_count
    BASELINE_ANGLE = mean_angle
    return BASELINE_ANGLE
######################################################################

def find_wordSpacing(image):
    WORD_SPACING = 0.0
    img = image.copy()
    filtered = bilateralFilter(img, 5, 50)

    thresh = threshold(filtered, 175)

    # extract a python list containing values of the horizontal projection of the image into 'hp'
    hpList = horizontalProjection(thresh)

    # Extracting 'Top Margin' feature.
    topMarginCount = 0
    for sum in hpList:
        # sum can be strictly 0 as well.
        if(sum<=255):
            topMarginCount += 1
        else:
            break

    lineTop = 0
    lineBottom = 0
    spaceTop = 0
    spaceBottom = 0
    setLineTop = True
    setSpaceTop = True
    includeNextSpace = True
    space_zero = [] # stores the amount of space between lines
    lines = [] # a 2D list storing the vertical start index and end index of each contour

    #  FIRST we extract the straightened contours from the image by looking at occurance of 0's in the horizontal projection.
    # we are scanning the whole horizontal projection now 
    for i, sum in enumerate(hpList):
        # sum being 0 means blank space
        if(sum<3000):
            if(setSpaceTop):
                spaceTop = i;
                setSpaceTop = False # spaceTop will be set once for each start of a space between lines
            spaceBottom = i+1;
        
            if(i<len(hpList)-1): # this condition is necessary to avoid array index out of bound error
                if(hpList[i+1]<3000): # if the next horizontal projectin is 0, keep on counting, it's still in blank space
                    continue
            # we are using this condition if the previous contour is very thin and possibly not a line
            if(includeNextSpace):
                space_zero.append(spaceBottom-spaceTop)
            else:
                if (len(space_zero)==0):
                    previous = 0
                else:
                    previous = space_zero.pop()
                space_zero.append(previous + spaceBottom-lineTop)
            setSpaceTop = True # next time we encounter 0, it's begining of another space so we set new spaceTop

        # sum greater than 0 means contour
        if(sum>3000):
            if(setLineTop):
                lineTop = i
                setLineTop = False # lineTop will be set once for each start of a new line/contour
            lineBottom = i+1

            if(i<len(hpList)-1): # this condition is necessary to avoid array index out of bound error
                if(hpList[i+1]>3000): # if the next horizontal projectin is > 0, keep on counting, it's still in contour
                    continue

                # if the line/contour is too thin <10 pixels (arbitrary) in height, we ignore it.
                # Also, we add the space following this and this contour itself to the previous space to form a bigger space: spaceBottom-lineTop.
                if(lineBottom-lineTop<20):
                    includeNextSpace = False
                    setLineTop = True # next time we encounter value > 0, it's begining of another line/contour so we set new lineTop
                    continue
            includeNextSpace = True # the line/contour is accepted, new space following it will be accepted

            # append the top and bottom horizontal indices of the line/contour in 'lines'
            lines.append([lineTop, lineBottom])
            setLineTop = True # next time we encounter value > 0, it's begining of another line/contour so we set new lineTop


    THRESHOLD = 15000
    new_space_row_count = 0
    total_row_count = 0
    total_lines_count = 0
    flag = False
    for i, line in enumerate(lines):
        segment = hpList[line[0]:line[1]]
        for j, sum in enumerate(segment):
            if(sum<THRESHOLD):
                new_space_row_count += 1
            else:
                total_row_count += 1
                flag = True

        # This line has contributed at least one count of pixel row of midzone
        if(flag):
            total_lines_count += 1
            flag = False

    # error prevention
    if(total_lines_count == 0): total_lines_count = 1

    total_space_row_count = new_space_row_count + np.sum(space_zero[1:-1]) #excluding first and last entries: Top and Bottom margins
    # the number of spaces is 1 less than number of lines but total_space_row_count contains the top and bottom spaces of the line
    average_line_spacing = float(total_space_row_count) / total_lines_count 
    average_letter_size = float(total_row_count) / total_lines_count
    # letter size is actually height of the letter and we are not considering width
    LETTER_SIZE = average_letter_size
    # error prevention 
    if(average_letter_size == 0): average_letter_size = 1
    # We can't just take the average_line_spacing as a feature directly. We must take the average_line_spacing relative to average_letter_size.
    # Let's take the ratio of average_line_spacing to average_letter_size as the LINE SPACING, which is perspective to average_letter_size.
    relative_line_spacing = average_line_spacing / average_letter_size
    LINE_SPACING = relative_line_spacing

    # Top marging is also taken relative to average letter size of the handwritting
    # relative_top_margin = float(topMarginCount) / average_letter_size
    # TOP_MARGIN = relative_top_margin

    width = thresh.shape[1]
    space_zero = [] # stores the amount of space between words
    # a 2D list storing the coordinates of each word: y1, y2, x1, x2

    # Isolated words or components will be extacted from each line by looking at occurance of 0's in its vertical projection.
    for i, line in enumerate(lines):
        extract = thresh[line[0]:line[1], 0:width] # y1:y2, x1:x2
        vp = verticalProjection(extract)
        wordStart = 0
        wordEnd = 0
        spaceStart = 0
        spaceEnd = 0
        setWordStart = True
        setSpaceStart = True
        includeNextSpace = False
        
    #     spaces = []

        # we are scanning the vertical projection
        for j, sum in enumerate(vp):
            # sum being 0 means blank space
            if(sum==0):
                if(setSpaceStart):
                    spaceStart = j
                    setSpaceStart = False # spaceStart will be set once for each start of a space between lines
                spaceEnd = j+1
                if(j<len(vp)-1): # this condition is necessary to avoid array index out of bound error
                    if(vp[j+1]==0): # if the next vertical projectin is 0, keep on counting, it's still in blank space
                        continue

                # we ignore spaces which is smaller than half the average letter size
                if((spaceEnd-spaceStart) > int(LETTER_SIZE/2)):
                    space_zero.append(spaceEnd-spaceStart)
                else:
                    includeNextSpace=True  
                    
                setSpaceStart = True # next time we encounter 0, it's begining of another space so we set new spaceStart

            # sum greater than 0 means word
            if(sum>0):
                if(setWordStart):
                    wordStart = j
                    setWordStart = False # wordStart will be set once for each start of a new word/component
                wordEnd = j+1
                if(j<len(vp)-1): # this condition is necessary to avoid array index out of bound error
                    if(vp[j+1]>0): # if the next horizontal projectin is > 0, keep on counting, it's still in non-space zone
                        continue

                # append the coordinates of each word/component: y1, y2, x1, x2 in 'words'
                # we ignore the ones which has height smaller than half the average letter size
                # this will remove full stops and commas as an individual component
                count = 0
                for k in range(line[1]-line[0]):
                    row = thresh[line[0]+k:line[0]+k+1, wordStart:wordEnd] # y1:y2, x1:x2
                    if(np.sum(row)):
                        count += 1
                    #U.S.A. special case not handelled
                if(count > int(LETTER_SIZE/2)):
                    if(includeNextSpace):
                        includeNextSpace = False
                        if(len(words)!=0):
                            words[len(words)-1][3] = wordEnd
                    else:
                        words.append([line[0], line[1], wordStart, wordEnd])

                setWordStart = True # next time we encounter value > 0, it's begining of another word/component so we set new wordStart

    #     space_zero.extend(spaces[1:-1])

    #print space_zero
    space_columns = np.sum(space_zero)
    space_count = len(space_zero)
    if(space_count == 0):
        space_count = 1
    average_word_spacing = float(space_columns) / space_count
    relative_word_spacing = average_word_spacing / average_letter_size
    WORD_SPACING = relative_word_spacing
    return relative_word_spacing

############################################################################################################

def find_slant(image):
    SLANT_ANGLE = 0.0
    theta = [-0.785398, -0.523599, -0.261799, -0.0872665, 0.01, 0.0872665, 0.261799, 0.523599, 0.785398]
    img=image.copy()
    # Corresponding index of the biggest value in s_function will be the index of the most likely angle in 'theta'
    s_function = [0.0] * 9
    count_ = [0]*9

    # apply bilateral filter
    filtered = bilateralFilter(img, 5, 50)

    # convert to grayscale and binarize the image by INVERTED binary thresholding
    # it's better to clear unwanted dark areas at the document left edge and use a high threshold value to preserve more text pixels
    thresh = threshold(filtered, 180)
    #cv2.imshow('thresh', lthresh)

    # loop for each value of angle in theta
    for i, angle in enumerate(theta):
        s_temp = 0.0 # overall sum of the functions of all the columns of all the words!
        count = 0 # just counting the number of columns considered to contain a vertical stroke and thus contributing to s_temp

        #loop for each word
        for j, word in enumerate(words):
            original = thresh[word[0]:word[1], word[2]:word[3]] # y1:y2, x1:x2

            height = word[1]-word[0]
            width = word[3]-word[2]

            # the distance in pixel we will shift for affine transformation
            # it's divided by 2 because the uppermost point and the lowermost points are being equally shifted in opposite directions
            shift = (math.tan(angle) * height) / 2

            # the amount of extra space we need to add to the original image to preserve information
            # yes, this is adding more number of columns but the effect of this will be negligible
            pad_length = abs(int(shift))
            
            x=height
            y=width+pad_length*2
            if(x<=0 or y<=0):
                continue
            # create a new image that can perfectly hold the transformed and thus widened image
            blank_image = np.zeros((height,width+pad_length*2,3), np.uint8)
            new_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
            new_image[:, pad_length:width+pad_length] = original

            # points to consider for affine transformation
            (height, width) = new_image.shape[:2]
            x1 = width/2
            y1 = 0
            x2 = width/4
            y2 = height
            x3 = 3*width/4
            y3 = height

            pts1 = np.float32([[x1,y1],[x2,y2],[x3,y3]])
            pts2 = np.float32([[x1+shift,y1],[x2-shift,y2],[x3-shift,y3]])
            M = cv2.getAffineTransform(pts1,pts2)
            deslanted = cv2.warpAffine(new_image,M,(width,height))

            # find the vertical projection on the transformed image
            vp = verticalProjection(deslanted)

            # loop for each value of vertical projection, which is for each column in the word image
            for k, sum in enumerate(vp):
                # the columns is empty
                if(sum == 0):
                    continue

                # this is the number of foreground pixels in the column being considered
                num_fgpixel = sum / 255

                # if number of foreground pixels is less than onethird of total pixels, it is not a vertical stroke so we can ignore
                if(num_fgpixel < int(height/3)):
                    continue

                # the column itself is extracted, and flattened for easy operation
                column = deslanted[0:height, k:k+1]
                column = column.flatten()

                # now we are going to find the distance between topmost pixel and bottom-most pixel
                # l counts the number of empty pixels from top until and upto a foreground pixel is discovered
                for l, pixel in enumerate(column):
                    if(pixel==0):
                        continue
                    break
                # m counts the number of empty pixels from bottom until and upto a foreground pixel is discovered
                for m, pixel in enumerate(column[::-1]):
                    if(pixel==0):
                        continue
                    break

                # the distance is found as delta_y, I just followed the naming convention in the research paper I followed
                delta_y = height - (l+m)

                # please refer the research paper for more details of this function, anyway it's nothing tricky
                h_sq = (float(num_fgpixel)/delta_y)**2

                #We are multiplying by a factor of num_fgpixel/height to the above function to yeild better result
                # this will also somewhat negate the effect of adding more columns and different column counts in the transformed image of the same word
                h_wted = (h_sq * num_fgpixel) / height

            

                # add up the values from all the loops of ALL the columns of ALL the words in the image
                s_temp += h_wted

                count += 1

        s_function[i] = s_temp
        count_[i] = count

    # finding the largest value and corresponding index
    max_value = 0.0
    max_index = 4
    for index, value in enumerate(s_function):
        #print (str(index)+" "+str(value)+" "+str(count_[index])
        if(value > max_value):
            max_value = value
            max_index = index
    # We will add another value 9 manually to indicate irregular slant behaviour.
    # This will be seen as value 4 (no slant) but 2 corresponding angles of opposite sign will have very close values.
    #print(max_index)
    if(max_index == 0):
        angle = 45
        result =  " : Extremely right slanted"
    elif(max_index == 1):
        angle = 30
        result = " : Above average right slanted"
    elif(max_index == 2):
        angle = 15
        result = " : Average right slanted"
    elif(max_index == 3):
        angle = 5
        result = " : A little right slanted"
    elif(max_index == 5):
        angle = -5
        result = " : A little left slanted"
    elif(max_index == 6):
        angle = -15
        result = " : Average left slanted"
    elif(max_index == 7):
        angle = -30
        result = " : Above average left slanted"
    elif(max_index == 8):
        angle = -45
        result = " : Extremely left slanted"
    elif(max_index == 4):
        p = s_function[4] / (s_function[3]+1)
        q = s_function[4] / (s_function[5]+1)
        #print ('p='+str(p)+' q='+str(q)
        # the constants here are abritrary but I think suits the best
        if((p <= 1.2 and q <= 1.2) or (p > 1.4 and q > 1.4)):
            angle = 0
            result = " : No slant"
        elif((p <= 1.2 and q-p > 0.4) or (q <= 1.2 and p-q > 0.4)):
            angle = 0
            result = " : No slant"
        else:
            max_index = 9
            angle = 0
            result =  " : No slant"
    return angle



hsum=[]
def Horizontal_sum(thresh):
    h, w = thresh.shape[:]

    for x in range(h):
        temp=0
        for y in range(w):
            if(thresh[x][y] > 0):
                temp=temp+thresh[x][y]
        if temp!=0:
            hsum.append(temp)
    return hsum

################################################################
def find_tbar(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsum=[]
    h, w = img.shape[:]
    inverted = img

    for x in range(h):
      for y in range(w):
        inverted[x][y] = 255 - img[x][y]

    filtered = cv2.bilateralFilter(inverted, 5, 50, 50)

    # Threshold the image
    ret,img = cv2.threshold(filtered, 127, 255, 0)
    lst = Horizontal_sum(img)

    ind = -1
    maxx = 0
    for i, val in enumerate(lst):
        if val > maxx:
            maxx = val
            ind = i

    if(ind < len(lst)/10):
        return 1
    else:
        return 0

page = st.sidebar.selectbox("Predict & About Us", ("Predict", "About Us"))


#t-bar
fin_tbar=0
uploaded_file2 = st.file_uploader("Please Upload your Handwriting Sample containing only small letter t",type=['png','jpeg','jpg'])

if uploaded_file2 is not None:
    img2=np.array(Image.open(uploaded_file2))
    fin_tbar=find_tbar(img2)

features.append(fin_tbar)

uploaded_file = st.file_uploader("Please Upload your Handwriting Sample",type=['png','jpeg','jpg'])


if uploaded_file is not None:
    #file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
    #st.write(file_details)

    img = np.array(Image.open(uploaded_file))
    
    image=img.copy()

    #Pressure 
    fin_pressure=find_pressure(img)
    features.append(fin_pressure)

    #Baseline
    fin_baseline=find_baseline(img)
    features.append(fin_baseline)

    #Word Spacing
    fin_wordSpacing=find_wordSpacing(img)
    features.append(fin_wordSpacing)

    #Slant Angle 
    fin_slant=find_slant(img)
    features.append(fin_slant)
    st.write("Feature vector for the Handwriting sample is : ")
    st.write(features)

    #Model Training and Accuracy 


    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn import metrics

    df = pd.read_csv('personality_data.csv')



    df_neuro = df.loc[:,'t - bar':'slant']
    df_neuro['N'] = df['N']

    # Second Feature :- Openness to Experience
    df_open = df.loc[:,'t - bar':'slant']
    df_open['O'] = df['O']

    # Third Feature :- Extraversion
    df_extra =  df.loc[:,'t - bar':'slant']
    df_extra['E'] = df['E']

    # Forth Feature :- Agreeableness
    df_agree = df.loc[:,'t - bar':'slant']
    df_agree['A'] = df['A']

    # Fifth feature :- Conscientiousness
    df_consc = df.loc[:,'t - bar':'slant']
    df_consc['C'] = df['C']

    # Neutral Feature
    df_neutral = df.loc[:,'t - bar':'slant']
    df_neutral['neutral'] = df['neutral']

    #new_sample =[[0,5.99622622,1.714579264,1.727740986,-5]]
    new_sample=[features]

    from sklearn.ensemble import RandomForestClassifier

    # input/optput for model
    features_rf_neuro = df_neuro.loc[:,:'slant'].values
    target_rf_neuro = df_neuro.loc[:,'N'].values

    # spliting dataset
    xtrain, xtest, ytrain, ytest = train_test_split(features_rf_neuro, target_rf_neuro, test_size = 0.25,random_state=42)

    classifier = RandomForestClassifier(min_samples_split=2, min_samples_leaf=1, max_features='auto', 
                                    max_depth=90, n_estimators = 1300, random_state=42)
    classifier.fit(xtrain, ytrain)
    y_pred=classifier.predict(xtest)

    sample_pred_rf_neuro = classifier.predict(new_sample)


    #openness

    # input/optput for model
    features_rf_open = df_open.loc[:,:'slant'].values
    target_rf_open= df_open.loc[:,'O'].values

    # spliting dataset
    xtrain, xtest, ytrain, ytest = train_test_split(features_rf_open, target_rf_open, test_size = 0.25,random_state=42)

    classifier = RandomForestClassifier(min_samples_split=2, min_samples_leaf=1, max_features='auto', max_depth=90, n_estimators = 1300, random_state=42)
    classifier.fit(xtrain, ytrain)
    y_pred=classifier.predict(xtest)

    sample_pred_rf_open = classifier.predict(new_sample)


    #extraversion

    # input/optput for model
    features_rf_extra = df_extra.loc[:,:'slant'].values
    target_rf_extra= df_extra.loc[:,'E'].values

    # spliting dataset
    xtrain, xtest, ytrain, ytest = train_test_split(features_rf_extra, target_rf_extra, test_size = 0.25,random_state=42)

    classifier = RandomForestClassifier(min_samples_split=2, min_samples_leaf=1, max_features='auto', max_depth=90, n_estimators = 1300, random_state=42)
    classifier.fit(xtrain, ytrain)
    y_pred=classifier.predict(xtest)

    sample_pred_rf_extra= classifier.predict(new_sample)


    #agreeableness

    # input/optput for model
    features_rf_agree = df_agree.loc[:,:'slant'].values
    target_rf_agree= df_agree.loc[:,'A'].values

    # spliting dataset
    xtrain, xtest, ytrain, ytest = train_test_split(features_rf_agree, target_rf_agree, test_size = 0.25,random_state=42)

    classifier = RandomForestClassifier(min_samples_split=2, min_samples_leaf=1, max_features='auto',max_depth=90, n_estimators = 1300, random_state=42)
    classifier.fit(xtrain, ytrain)
    y_pred=classifier.predict(xtest)


    sample_pred_rf_agree = classifier.predict(new_sample)


    #Conscientiousness

    # input/optput for model
    features_rf_consc = df_agree.loc[:,:'slant'].values
    target_rf_consc= df_agree.loc[:,'A'].values

    # spliting dataset
    xtrain, xtest, ytrain, ytest = train_test_split(features_rf_consc, target_rf_consc, test_size = 0.25,random_state=42)

    classifier = RandomForestClassifier(min_samples_split=2, min_samples_leaf=1, max_features='auto', max_depth=90, n_estimators = 1300, random_state=42)
    classifier.fit(xtrain, ytrain)
    y_pred=classifier.predict(xtest)


    sample_pred_rf_consc= classifier.predict(new_sample)


    rf_output = [sample_pred_rf_neuro,sample_pred_rf_open,sample_pred_rf_extra,sample_pred_rf_agree,sample_pred_rf_consc] 

    st.write(" ")
    st.write("Prediction Vector : ")
    st.write(" ")
    if sample_pred_rf_neuro==1:
        st.write("1 => Neuroticism")
    else :
        st.write("0 => Neuroticism")


    if sample_pred_rf_open==1:
        st.write("1 => Openness to experience")
    else :
        st.write("0 => Openness to experience")


    if sample_pred_rf_extra==1:
        st.write("1 => Extraversion")
    else :
        st.write("0 => Extraversion")

    if sample_pred_rf_agree==1:
        st.write("1 => Agreeableness")
    else :
        st.write("0 => Agreeableness")

    if sample_pred_rf_consc==1:
        st.write("1 => Conscientiousness")
    else :
        st.write("0 => Conscientiousness")
        
    st.write(" ")
    st.write(" ")
    st.write("Your Personality Traits are as follows : ")

    if sample_pred_rf_neuro==1:
        st.write("1. You have a tendency to be 'nervous' rather than confident")
    else :
        st.write("1. You have a tendency to be 'confident' rather than nervous.")


    if sample_pred_rf_open==1:
        st.write("2. You have a 'curious' personality and have a desire to adventure")
    else :
        st.write("2. You have a 'curious' personality and have a desire to adventure")


    if sample_pred_rf_extra==1:
        st.write("3. You are generally 'outgoing' and talkative.")
    else :
        st.write("3. You remain 'solitary' and mostly choose not to talk.")

    if sample_pred_rf_agree==1:
        st.write("4. You are 'compassionate' instead of suspicious and have a forth coming attitude.")
    else :
        st.write("4. You are more 'suspicious' than compassionate.")

    if sample_pred_rf_consc==1:
        st.write("5. You are very 'organized' and believe in carefully planned.")
    else :
        st.write("5. You are sometimes 'careless' and less organized.")


    
    











