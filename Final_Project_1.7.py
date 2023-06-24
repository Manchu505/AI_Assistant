import cv2
import face_recognition
import speech_recognition as sr
import openai
from bardapi import Bard
import os
import textwrap
import config
import datetime
import grpc
import time
import numpy as np
import soundfile
from H3X_Assistant.system.grpc import audio2face_pb2_grpc
from H3X_Assistant.system.grpc import audio2face_pb2
import edge_tts

# Initialize ChatGPT API & sets the AI tone
openai.api_key = config.OPENAI_API_KEY
messages = []
assistant = "A helpful and friendly computer AI"
messages.append({"role": "system", "content": assistant})
# Initialize Google Bard API
os.environ['_BARD_API_KEY'] = config.Google_TOKEN



# From line 29- 114 is code used to drive the NVIDIA Omniverse Audio2Face - this code is taken directly from the Streaming Media folder for Omniverse.
def push_audio_track(url, audio_data, samplerate, instance_name):
    block_until_playback_is_finished = True  # ADJUST
    with grpc.insecure_channel(url) as channel:
        stub = audio2face_pb2_grpc.Audio2FaceStub(channel)
        request = audio2face_pb2.PushAudioRequest()
        request.audio_data = audio_data.astype(np.float32).tobytes()
        request.samplerate = samplerate
        request.instance_name = instance_name
        request.block_until_playback_is_finished = block_until_playback_is_finished
        print("Sending audio data...")
        response = stub.PushAudio(request)
        if response.success:
            print("SUCCESS")
        else:
            print(f"ERROR: {response.message}")
    print("Closed channel")
    # voice_input()


def push_audio_track_stream(url, audio_data, samplerate, instance_name):
    chunk_size = samplerate // 10  # ADJUST
    sleep_between_chunks = 0.04  # ADJUST
    block_until_playback_is_finished = True  # ADJUST

    with grpc.insecure_channel(url) as channel:
        print("Channel created")
        stub = audio2face_pb2_grpc.Audio2FaceStub(channel)

        def make_generator():
            start_marker = audio2face_pb2.PushAudioRequestStart(
                samplerate=samplerate,
                instance_name=instance_name,
                block_until_playback_is_finished=block_until_playback_is_finished,
            )
            # At first, we send a message with start_marker
            yield audio2face_pb2.PushAudioStreamRequest(start_marker=start_marker)
            # Then we send messages with audio_data
            for i in range(len(audio_data) // chunk_size + 1):
                time.sleep(sleep_between_chunks)
                chunk = audio_data[i * chunk_size: i * chunk_size + chunk_size]
                yield audio2face_pb2.PushAudioStreamRequest(audio_data=chunk.astype(np.float32).tobytes())

        request_generator = make_generator()
        print("Sending audio data...")
        response = stub.PushAudioStream(request_generator)
        if response.success:
            print("SUCCESS")
        else:
            print(f"ERROR: {response.message}")
    print("Channel closed")
    # voice_input()


def a2f():

    # Sleep time emulates long latency of the request
    # sleep_time = 2.0  # ADJUST

    # URL of the Audio2Face Streaming Audio Player server (where A2F App is running)
    url = "localhost:50051"  # ADJUST

    # Local input WAV file path
    # audio_fpath = sys.argv[1]
    audio_fpath = "data.mp3"

    # Prim path of the Audio2Face Streaming Audio Player on the stage (were to push the audio data)

    # instance_name = sys.argv[2]
    instance_name = "/World/LazyGraph/PlayerStreaming"

    data, samplerate = soundfile.read(audio_fpath, dtype="float32")

    # Only Mono audio is supported
    if len(data.shape) > 1:
        data = np.average(data, axis=1)

    # print(f"Sleeping for {sleep_time} seconds")
    # time.sleep(sleep_time)

    if 0:  # ADJUST
        # Push the whole audio track at once
        push_audio_track(url, data, samplerate, instance_name)
    else:
        # Emulate audio stream and push audio chunks sequentially
        push_audio_track_stream(url, data, samplerate, instance_name)

# Text to Speech engine using Microsoft Edge-TTS - and sends audio file to Audio2Face - limit returns from Google to text only.
def speak(text) -> None:
    ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!-_$: ")
    clean_text = ''.join(c for c in text if c in ALLOWED_CHARS)
    voice = 'en-US-ChristopherNeural'
    command = f'edge-tts --voice "{voice}" --text "{clean_text}" --write-media "data.mp3" --write-subtitles=x.vtt'
    os.system(command)
    time.sleep(1.0)
    a2f()

# After succesful facial recognition the system will greet the user based on time of day
def welcome():
    global welcome
    hour = datetime.datetime.now().hour
    if 0 <= hour < 12:
        text = "Good Morning." + name + " welcome to  H3X. Please choose either Google or Open-AI to begin............................................"
    elif 12 <= hour < 18:
        text = "Good Afternoon." + name + " welcome to  H3X. Please choose either Google or Open-AI to begin............................................"
    else:
        text = "Good Evening." + name + " welcome to  H3X. Please choose either Google or Open-AI to begin............................................"

    speak(text)
    chat()

# if facial recognition is unsuccessful - loops back to facial recognition.
def unrecognized():
    now = datetime.datetime.now()
    hour = now.hour
    print('I am sorry I do not recognised you!')
    if hour < 12:
        speak(
            "good morning welcome to HEX SHIELD, I am sorry I do not recognize you -- Please press any key to try again.  ")
    elif hour < 18:
        speak(
            "good afternoon welcome to HEX SHIELD I am sorry I do not recognize you -- Please press any key to try again.  ")
    else:
        speak(
            "good evening welcome to HEX SHIELD I am sorry I do not recognize you -- Please press any key to try again.  ")
    os.system('pause')
    # main()

# Closes the python interface and cleans up files that were used during audio push.
def shut_down():
    speak("Shutting down, Thank you for using H3X    " + name)
    time.sleep(5)
    os.remove("data.mp3")
    os.remove("x.vtt")
    exit(0)

# Restarts the program to bring in new user if needed.
def start_over():
    speak("Starting over" + "  goodbye  " + name)
    time.sleep(5)
    main()

# Sends voice input to OpenAI ChatGPT and returns answer
def chatGPT():
    speak("Welcome to Open-AI's ChatGPT!")
    while True:
        rec = sr.Recognizer()
        with sr.Microphone() as source:
            rec.adjust_for_ambient_noise(source)
            print("\nListening ..................................")
            audio = rec.listen(source)
        try:
            chatGPT_message = rec.recognize_google(audio, language="en-US")
            # Exits Program
            if chatGPT_message.lower() == "exit":
                print("\nGoodbye!")
                shut_down()
            # Starts program over
            elif chatGPT_message.lower() == "start over":
                print("Starting Over?")
                start_over()
            # Switches to Google Bard
            elif chatGPT_message.lower() == "google":
                print("Switching to Google Bard")
                bard()

            else:
                print("User: " + chatGPT_message)
                print("ChatGPT is processing  ")
                messages.append({"role": "user", "content": chatGPT_message})
                chat = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.5,
                    max_tokens=500,
                )
                reply = chat.choices[0].message.content
                messages.append({"role": "assistant", "content": reply})
                print("\nChatBot : ---------------------------------------------\n")
                wrapper = textwrap.TextWrapper(width=80)
                replyOutput = wrapper.wrap(text=reply)
                for element in replyOutput:
                    print(element)
                speak(reply)

        except sr.UnknownValueError:
            print("Please say that again ")

        except sr.RequestError:
            print("Could not request results from Google Recognizer")

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            shut_down()

# Sends voice input to Google Bard and returns answer
def bard():
    speak("Welcome to Google Bard!")
    while True:
        rec = sr.Recognizer()
        with sr.Microphone() as source:
            rec.adjust_for_ambient_noise(source)
            print("\nListening ..................................")
            audio = rec.listen(source)
        try:
            google_message = rec.recognize_google(audio, language="en-US")
            # Exits program
            if google_message.lower() == "exit":
                print("\nGoodbye!")
                shut_down()
            # Starts program over
            elif google_message.lower() == "start over":
                print("Starting Over?")
                start_over()
            # Switches to OpenAI
            elif google_message.lower() == "open":
                print("Switching to ChatGPT")
                chatGPT()

            else:
                print("Google Bard is processing " + google_message + " \n")
                bard_reply = (Bard().get_answer(google_message)['content'])
                print("\n Google Bard: ----------------------------------------------\n")
                wrapper = textwrap.TextWrapper(width=80)
                replyOutput = wrapper.wrap(text=bard_reply)
                for element in replyOutput:
                    print(element)
                speak(bard_reply)

        except sr.UnknownValueError:
            print("Please say that again ")

        except sr.RequestError:
            print("Could not request results from Google Recognizer")

# Starts the Chat cycle - allowing different paths to take based on what user want to do.
def chat():
    while True:
        # speak("Please choose either Google or Open-AI to begin............................................")
        rec = sr.Recognizer()
        with sr.Microphone() as source:
            rec.adjust_for_ambient_noise(source)
            # rec.pause_threshold = 3
            print("\nListening ..................................")
            audio = rec.listen(source)
        try:
            option = rec.recognize_google(audio, language="en-US")
            # Exits program
            if option.lower() == "exit":
                print("\nGoodbye!")
                shut_down()
            # Starts the program over
            elif option.lower() == "start over":
                print("Starting Over?")
                start_over()
            # Starts OpenAI
            elif option.lower() == "open":
                print("Switching to ChatGPT")
                chatGPT()
            #  Starts Google Bard
            elif option.lower() == "google":
                print("Switching to Google Bard")
                bard()

        except sr.UnknownValueError:
            print("Please say that again ")

        except sr.RequestError:
            print("Could not request results from Google Recognizer")

# Facial recognition based on photos of known users.
def main():
    global name
    # Load a sample picture and learn how to recognize it.
    grant_image = face_recognition.load_image_file("dataset/Grant.jpg")
    grant_face_encoding = face_recognition.face_encodings(grant_image)[0]

    hayley_image = face_recognition.load_image_file("dataset/Hayley.jpeg")
    hayley_face_encoding = face_recognition.face_encodings(hayley_image)[0]

    darwin_image = face_recognition.load_image_file("dataset/Darwin.jpeg")
    darwin_face_encoding = face_recognition.face_encodings(darwin_image)[0]

# Create arrays of known face encodings and their names
    known_face_encodings = [
        grant_face_encoding,
        hayley_face_encoding,
        darwin_face_encoding,
      ]
    known_face_names = [
        "Grant",
        "Hayley",
        "Darwin",
    ]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    name = " "

    capture = cv2.VideoCapture(0)

    while True:
        # global name
        # Grab a single frame of video
        ret, frame = capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 4, bottom - 4), font, 1, (0, 0, 255), 2)
            if name != "Unknown":
                capture.release()
                cv2.destroyAllWindows()
                welcome()
            else:
                unrecognized()
        # Display the resulting image
        cv2.imshow('Image', frame)
        # if name != "Unknown":
        #     print("Hello  ")

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
