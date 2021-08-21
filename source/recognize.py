import cv2
import os
import mobilenet as mn
import math


def prepare_video(capture_realtime, video_path, video_filename):
    if capture_realtime:
        # read vd from camera
        video = cv2.VideoCapture(0)
        video.set(3, 640)
        video.set(4, 480)
        output_file = '../output/Streaming.mp4'
    else:
        # read vd from mp4 file
        video = cv2.VideoCapture(video_path + video_filename)
        output_file = '../output/' + video_filename + '_detect_ALL.mp4'
    # Check if camera opened successfully
    if not video.isOpened():
        print("Error opening video stream or file")
        exit()
    return video, output_file


def get_video_resolution(video):
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    print("video/image width: %d" % frame_width)
    print("video/image height: %d" % frame_height)
    return frame_width, frame_height


# size can be defined as: area, diagonal, or maximum of width and height ...
def calculate_size(box):
    # box is of type Rect_ (_Tp _x, _Tp _y, _Tp _width, _Tp _height)
    # for example box = {ndarray [1061, 868, 78, 36]}
    return calculate_area(box)


def calculate_area(box):
    return box[2] * box[3]


def calculate_diagonal(box):
    return math.sqrt(pow(box[2], 2) + pow(box[3], 2))


def calculate_max(box):
    if box[2] > box[3]:
        return box[2]
    return box[3]


def calc_distance(size):
    # we fit a linear model to predict distance based on stop sign size;
    # intercept and slope are two parameters of the model, calibrated by
    # measuring actual size and distance samples with my own camera;
    # to predict using a autonomous vehicle, need to measure with its own
    # camera, collect data, and re-calibrate the parameters
    intercept = 8.0932
    slope = 173320
    return intercept + slope * (1.0 / size)


def process_video(net, video, out, capture_realtime, detection_threshold, class_names, detect_class_filter):
    count = 0
    # Read until video is completed
    while video.isOpened():
        # Capture frame-by-frame
        success, frame = video.read()
        if success:
            class_ids, confs, bbox = net.detect(frame, confThreshold=detection_threshold)
            print("Processing frame%d:" % count)
            # print(class_ids, bbox)

            if len(class_ids) > 0:
                for classId, confidence, box in zip(class_ids.flatten(), confs.flatten(), bbox):
                    class_name = class_names[classId - 1].upper()
                    if (not detect_class_filter) or (class_name in detect_class_filter):
                        print(class_name, box)
                        size = calculate_size(box)
                        print("box size = " + str(size))
                        print("predicted distance = " + str(calc_distance(size)) + " cm ")
                        # add code to publish (distance, confidence, size) to ROS workspace/topic here:
                        # ......
                        cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                        # add names to box origin + (10,30)
                        cv2.putText(frame, class_name, (box[0] + 10, box[1] + 30),
                                    cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
                        # add conf level to box origin + (10,60)
                        cv2.putText(frame, str(round(confidence * 100, 2)) + '%', (box[0] + 10, box[1] + 60),
                                    cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)

            # Display the resulting frame
            if not capture_realtime:
                out.write(frame)
                cv2.imwrite("../output/frame%d.jpg" % count, frame)
            # display frame after writing to file(s)
            cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break

        count += 1


def clean_up(video, out):
    # When everything done, release the video capture object
    video.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()


def main():
    print(os.getcwd())
    video_path = '../videos/'
    video_filename = 'Stop sign recognition.mp4'

    # video_path = '../videos/Samsung Galaxy G20/'
    video_filenames = [
        '20210705_145202.mp4',
        '20210705_145450.mp4',
        '20210705_150729.mp4',
        '20210705_153646.mp4'
    ]
    # video_filename = video_filenames[1]

    detection_threshold = 0.5
    # proecess streaming video from system's default camera if capture_realtime is set to True;
    # otherwise process recorded video file located at video_path/vedeo_filename
    capture_realtime = True
    detect_class_filter = ['STOP SIGN']

    class_names = mn.get_class_names()
    net = mn.init_model_network()
    vd, output_file = prepare_video(capture_realtime, video_path, video_filename)
    # Define the codec and create VideoWriter object
    # out = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M','J','P','G'), 10, get_video_resolution(vd))
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 10, get_video_resolution(vd))
    process_video(net, vd, out, capture_realtime, detection_threshold, class_names, detect_class_filter)
    clean_up(vd, out)


if __name__ == "__main__":
    main()
