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
    train_path = "[PATH_TO_TRAIN_DIRECTORY]"
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
    test_path = "[PATH_TO_TEST_ROOT_DIRECTORY]"
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