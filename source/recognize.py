import cv2
import os
import mobilenet as mn


def prepare_video(capture_realtime):
    if capture_realtime:
        # read vd from camera
        video = cv2.VideoCapture(0)
        video.set(3, 640)
        video.set(4, 480)
        output_file = '../output/Streaming.mp4'
    else:
        # read vd from mp4 file
        video = cv2.VideoCapture('../videos/Stop sign recognition.mp4')
        output_file = '../output/Stop sign recognition_detect_ALL.mp4'
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


def process_video(net, video, out, capture_realtime, detection_threshold, class_names):
    count = 0
    # Read until video is completed
    while video.isOpened():
        # Capture frame-by-frame
        success, frame = video.read()
        if success:
            class_ids, confs, bbox = net.detect(frame, confThreshold=detection_threshold)
            print(class_ids, bbox)

            if len(class_ids) > 0:
                for classId, confidence, box in zip(class_ids.flatten(), confs.flatten(), bbox):
                    print(class_names[classId - 1].upper(), box)
                    cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                    # add names to box origin + (10,30)
                    cv2.putText(frame, class_names[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
                    # add conf level to box origin + (10,60)
                    cv2.putText(frame, str(round(confidence * 100, 2)) + '%', (box[0] + 10, box[1] + 60),
                                cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)

            # Display the resulting frame
            if not capture_realtime:
                out.write(frame)
                cv2.imwrite("../output/frame%d.jpg" % count, frame)
            # display frame after writing to file(s)
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
    detection_threshold = 0.7
    capture_realtime = False
    class_names = mn.get_class_names()
    net = mn.init_model_network()
    vd, output_file = prepare_video(capture_realtime)
    # Define the codec and create VideoWriter object
    # out = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M','J','P','G'), 10, get_video_resolution(vd))
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 10, get_video_resolution(vd))
    process_video(net, vd, out, capture_realtime, detection_threshold, class_names)
    clean_up(vd, out)


if __name__ == "__main__":
    main()