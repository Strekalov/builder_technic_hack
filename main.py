import streamlit as st
import warnings
import time
import json
from statistics_tables import get_tables

from .inference.gigatrack import run

warnings.filterwarnings("ignore")

st.header("Видеоаналитика на строительных объектах")

tab1, tab2 = st.tabs(["Присутствие/отсутствие в кадре", "Работа/простой"])
#print(video_file)
#video_path = "out.mp4"
video_path2 = "output.webm"
old_json_path = "old_json.json"
new_json_path = "new_json.json"

json_file = json.load(open(old_json_path))

with tab1:
    #video_path = "~/projects/fire1.mp4"


    try:
        from streamlit_terran_timeline import terran_timeline, generate_timeline
    except ImportError:
        warnings.warn(
            "Failed to load terran_timeline from streamlit_terran_timeline. "
            "Please run 'pip install streamlit_terran_timeline' or "
            "'pip install .' if working locally"
        )
        exit(1)


    st.subheader("Загрузите своё видео")

    video_file = st.file_uploader("Выберите видеофайл для анализа присутствия техники на стройке", type=(["mp4"]))
    if video_file is not None:
        video_path = video_file.name
        print(video_path)
    else:
        video_path = "output.mp4"

    #video_path = st.text_input(
    #    "Путь к видео для анализа присутствия техники на стройке", video_path
    #)

    #
    # Show the actual faces timeline chart
    #
    st.subheader("Таймлайн работы техники")
    st.write("")
    st.write("(Зелёный цвет обозначает присутствие единицы техники в кадре, а оранжевый - её отсутствие)")

    st.cache_data.clear()
    st.cache_resource.clear()
    @st.cache(persist=True, ttl=86_400, suppress_st_warning=True, show_spinner=False)
    #@st.cache_resource()
    def _generate_timeline(video_path, json_file):
        timeline = generate_timeline(
            video_src=video_path,
            json_f=json_file,
            appearence_threshold=5,
            batch_size=16,
            duration=None,
            framerate=6,
            output_directory="timelines",
            ref_directory=None,
            similarity_threshold=0.75,
            start_time=0,
            thumbnail_rate=1,
        )

        return timeline


    with st.spinner("Генерируем таймлайн"):
        #timeline = generate_timeline(video_path, batch_size=8)
        
        args = {
            "conf": 0.8,
            "source": video_path,
            }

        json1_path, json2_path, videooutput_path2 = run(**args)

        timeline = _generate_timeline(video_path, json_file)

    start_time = terran_timeline(timeline, key='timeline_tab1')

    #print(start_time)

    seconds = start_time
    minutes = seconds // 60
    hours = minutes // 60

    #print "%02d:%02d:%02d" % (hours, minutes % 60, seconds % 60)

    st.write(f"Выбрано время: %02d:%02d:%02d" % (hours, minutes % 60, seconds % 60))
    #placeholder1 = st.empty()
    #placeholder2 = st.empty()
    col1, col2 = st.columns(2)
    col1.header("Original")
    col2.header("With boxes")

    #placeholder1 = st.empty()
    #placeholder2 = st.empty()
    print("start_time", start_time)

        # placeholder2 = st.empty()
        # print('col2 start_time', start_time)
        # placeholder2.video(video_path2, start_time=int(start_time))

with col1:
    placeholder1 = st.empty()
    print('col1 start_time', start_time)
    placeholder1.video(video_path, start_time=int(start_time))

with col2:
    placeholder2 = st.empty()
    print('col2 start_time', start_time)
    placeholder2.video(video_path2, start_time=int(start_time))


with tab2:
    # video_path = "~/projects/fire1.mp4"

    try:
        from streamlit_terran_timeline import terran_timeline, generate_timeline
    except ImportError:
        warnings.warn(
            "Failed to load terran_timeline from streamlit_terran_timeline. "
            "Please run 'pip install streamlit_terran_timeline' or "
            "'pip install .' if working locally"
        )
        exit(1)

    st.subheader("Загрузите своё видео")

    video_file = st.file_uploader("Выберите видеофайл для анализа работы и простоя техники", type=(["mp4"]))
    if video_file is not None:
        video_path = video_file.name
        print(video_path)
    else:
        video_path = "output.mp4"

    #video_path = st.text_input(
    #    "Путь к видео для анализа работы и простоя техники", video_path
    #)


    st.subheader("Таймлайн работы техники")
    st.write("")
    st.write("(Зелёный цвет обозначает время работы единицы техники, а оранжевый - время её простоя)")

    st.cache_data.clear()
    st.cache_resource.clear()


    @st.cache(persist=True, ttl=86_400, suppress_st_warning=True, show_spinner=False)
    # @st.cache_resource()
    def _generate_timeline(video_path, json_file):
        timeline = generate_timeline(
            video_src=video_path,
            json_f=json_file,
            appearence_threshold=5,
            batch_size=16,
            duration=None,
            framerate=6,
            output_directory="timelines",
            ref_directory=None,
            similarity_threshold=0.75,
            start_time=0,
            thumbnail_rate=1,
        )

        return timeline


    with st.spinner("Генерируем таймлайн"):
        # timeline = generate_timeline(video_path, batch_size=8)
        timeline = _generate_timeline(video_path, json_file)

    start_time = terran_timeline(timeline, key='timeline_tab2')

    # print(start_time)

    seconds = start_time
    minutes = seconds // 60
    hours = minutes // 60

    # print "%02d:%02d:%02d" % (hours, minutes % 60, seconds % 60)

    st.write(f"Выбрано время: %02d:%02d:%02d" % (hours, minutes % 60, seconds % 60))
    placeholder1 = st.empty()
    placeholder2 = st.empty()
    col1, col2 = st.columns(2)
    placeholder1 = st.empty()
    placeholder2 = st.empty()
    print("start_time", start_time)




with col1:
    placeholder1 = st.empty()
    print('col1 start_time', start_time)
    placeholder1.video(video_path, start_time=int(start_time))

with col2:
    placeholder2 = st.empty()
    print('col2 start_time', start_time)
    placeholder2.video(video_path2, start_time=int(start_time))


st.write(" ")

with open("newdata.json", "r") as f:
    data = json.loads(f.read())
    st.table(get_tables(data))
