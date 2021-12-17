import os
import cv2
import numpy as np

def get_train_image(path):
    '''
        To get a list of train images, images label, and images index using the given path

        Parameters
        ----------
        path : str
            Location of train root directory
        
        Returns
        -------
        list
            List containing all the train images
        list
            List containing all train images label
        list
            List containing all train images indexes
    '''
    image_list = []
    image_index = []
  
    train_directory = os.listdir(path)
    for index, train in enumerate(train_directory):
        image_path_list = os.listdir(path + '/' + train)
        image_list.append(train)
        for image_path in image_path_list:
            if(image_path[-3:]=='jpg'):
                image_index.append(index)
    image_data = []
    train_directory = os.listdir(path)
    for train in train_directory:
        image_path_list = os.listdir(path + '/' + train)
        for image_path in image_path_list:
                if(image_path[-3:]=='jpg'):
                    image = cv2.imread(path +'/' + train + '/' + image_path)
                    image_data.append(image)
    return image_data,image_list,image_index


def get_all_test_folders(path):
    '''
        To get a list of test subdirectories using the given path

        Parameters
        ----------
        path : str
            Location of test root directory
        
        Returns
        -------
        list
            List containing all the test subdirectories
    '''
    folders = os.listdir(path)
    return folders


def get_all_test_images(path):
    '''
        To load a list of test images from given path list. Resize image height 
        to 200 pixels and image width to the corresponding ratio for train images

        Parameters
        ----------
        path : str
            Location of images root directory
        
        Returns
        -------
        list
            List containing all image that has been resized for each Test Folders
    '''
    image_data = [] 
    image_path_list = os.listdir(path + '/')
    for image_path in image_path_list:
        if(image_path[-3:]=='jpg'):
            image = cv2.imread(path +'/'+ image_path)
            width = 300
            height = 200
            dim = (width, height)
            resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            image_data.append(resized)
    return image_data


def detect_faces_and_filter(faces_list, labels_list=None):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is not equals to one

        Parameters
        ----------
        faces_list : list
            List containing all loaded images
        labels_list : list
            List containing all image classes labels
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            list containing image gray face location
        list
            List containing all filtered image classes label
    '''
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    image_location = []
    image_penampung = []
    label_penampung = []
    for idx,image in enumerate(faces_list):
        faces = face_cascade.detectMultiScale(image, 1.2, 3)
        if len(faces)==1:
            for x,y,w,h in faces:
                image_location.append((x,y,w,h))
                image_crop = image[y:y+h,x:x+w]
                width = image_crop.shape[1]
                height = image_crop.shape[0]
                dim = (width,height)
                image_crop = cv2.resize(image_crop,dim)
                image_crop = cv2.cvtColor(image_crop,cv2.COLOR_BGR2GRAY)
                image_penampung.append(image_crop)
                if labels_list is not None:
                    label_penampung.append(labels_list[idx])
    return image_penampung,image_location,label_penampung


def train(grayed_images_list, grayed_labels_list):
    '''
        To create and train face recognizer object

        Parameters
        ----------
        grayed_images_list : list
            List containing all filtered and cropped face images in grayscale
        grayed_labels : list
            List containing all filtered image classes label
        
        Returns
        -------
        object
            Recognizer object after being trained with cropped face images
    '''
    lbph = cv2.face.LBPHFaceRecognizer_create()
    lbph.train(grayed_images_list,np.array(grayed_labels_list))
    return lbph


def predict(recognizer, gray_test_image_list):
    '''
        To predict the test image with the recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        gray_test_image_list : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''
    prediction_list = []
    for image in gray_test_image_list:
        lists, _ = recognizer.predict(image)
        prediction_list.append(lists)
    return prediction_list


def check_attandee(predicted_name, room_number):
    '''
        To check the predicted user is in the designed room or not

        Parameters
        ----------
        predicted_name : str
            The name result from predicted user
        room_number : int
            The room number that the predicted user entered

        Returns
        -------
        bool
            If predicted user entered the correct room return True otherwise False
    '''
    verified_names_room_1 = ['Elon Musk', 'Steve Jobs', 'Benedict Cumberbatch','Donald Trump']
    verified_names_room_2 = ['IU', 'Kim Se Jeong', 'Kim Seon Ho', 'Rich Brian']
    if room_number == 1:
        if predicted_name in verified_names_room_1:
            return True
        else:
            return False
    elif room_number == 2:
        if predicted_name in verified_names_room_2:
            return True
        else:
            return False
    
def write_prediction(predict_results, test_image_list, test_faces_rects, train_names, room):
    '''
        To draw prediction and validation results on the given test images

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories
        room: int
            The room number

        Returns
        -------
        list
            List containing all test images after being drawn with
            its prediction and validation results
    '''
    images = []
    for idx,image in enumerate(test_image_list):
        x, y, w, h = test_faces_rects[idx]
        attandee_check = check_attandee(train_names[predict_results[idx]],room)
        if attandee_check == True:   
            image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
            image = cv2.putText(image,train_names[predict_results[idx]]+" - Present",(90,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,(0,255,0))
            images.append(image)
        else:
            image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
            image = cv2.putText(image,train_names[predict_results[idx]]+" - Shouldn't be here",(90,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,(0,0,255))
            images.append(image)
    return images


def combine_and_show_result(room, predicted_test_image_list):
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        room : str
            The room number in string format (e.g. 'Room 1')
        predicted_test_image_list : nparray
            Array containing image data
    '''
    combined_image = np.hstack(predicted_test_image_list)
    cv2.imshow(room,combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''

def main():
    
    '''
        Please modify train_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_path = "Dataset/Train"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    faces_list, labels_list, indexes_list = get_train_image(train_path)
    grayed_trained_images_list, _, grayed_trained_labels_list = detect_faces_and_filter(faces_list, indexes_list)
    recognizer = train(grayed_trained_images_list, grayed_trained_labels_list)

    '''
        Please modify test_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_path = "Dataset/Test"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_images_folder = get_all_test_folders(test_path)

    for index, room in enumerate(test_images_folder):
        test_images_list = get_all_test_images(test_path + '/' + room)
        grayed_test_image_list, grayed_test_location, _ = detect_faces_and_filter(test_images_list)
        predict_results = predict(recognizer, grayed_test_image_list)
        predicted_test_image_list = write_prediction(predict_results, test_images_list, grayed_test_location, labels_list, index+1)
        combine_and_show_result(room, predicted_test_image_list)


if __name__ == "__main__":
    main()