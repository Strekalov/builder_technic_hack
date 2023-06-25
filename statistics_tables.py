import json
from collections import defaultdict
from datetime import datetime
from time import gmtime, strftime

import pandas as pd
import streamlit as st
def calc(t_1):
    return strftime("%H:%M:%S", gmtime(t_1 // 60))


def get_tables(data: dict):
    class_work_time = defaultdict(int)
    total_work_time = 0
    total_excavator_work_time = 0
    total_crane_work_time = 0
    total_concrete_mixer_work_time = 0
    total_truck_work_time = 0
    total_tractor_work_time = 0
    total_asphalt_paver_work_time = 0

    for vals in data.values():
        for obj in vals:
            class_id = obj["technic_type_id"]
            work_time = obj["worked_end_timestamp"] - obj["worked_start_timestamp"]
            class_work_time[class_id] += work_time
            total_work_time += work_time

            if class_id == 1:
                total_excavator_work_time += work_time
            elif class_id == 2:
                total_crane_work_time += work_time
            elif class_id == 3:
                total_concrete_mixer_work_time += work_time
            elif class_id == 4:
                total_truck_work_time += work_time
            elif class_id == 5:
                total_tractor_work_time += work_time
            elif class_id == 6:
                total_asphalt_paver_work_time += work_time

    class_work_time_dict = dict(class_work_time)
    class_work_time_df = pd.DataFrame(
        list(class_work_time_dict.items()),
        columns=["Класс техники", "Время работы "],
    )

    class_work_time_df["Время работы "] = class_work_time_df[
        "Время работы "
    ].apply(calc)

    st.table(class_work_time_df)

    total_work_time_dict = {
        "Экскаватор": strftime("%H:%M:%S", gmtime(total_excavator_work_time // 60)),
        "Кран": strftime("%H:%M:%S", gmtime(total_crane_work_time // 60)),
        "Бетономешалка": strftime(
            "%H:%M:%S", gmtime(total_concrete_mixer_work_time // 60)
        ),
        "Грузовик": strftime("%H:%M:%S", gmtime(total_truck_work_time // 60)),
        "Трактор": strftime("%H:%M:%S", gmtime(total_tractor_work_time // 60)),
        "Асфальтоукладчик": strftime(
            "%H:%M:%S", gmtime(total_asphalt_paver_work_time // 60)
        ),
    }

    total_work_time_df = pd.DataFrame(
        total_work_time_dict.items(),
        columns=["Класс техники", "Общее время работы по всем классам"],
    )

    st.table(total_work_time_df)

    def calculate_work_time(object_data):
        start_time = datetime.fromtimestamp(object_data["worked_start_timestamp"])
        end_time = datetime.fromtimestamp(object_data["worked_end_timestamp"])
        work_time = (end_time - start_time).total_seconds() / 60
        return work_time

    objects_info = []

    for object_class in data.values():
        for object_data in object_class:
            work_time = calculate_work_time(object_data)
            objects_info.append(
                {
                    "ID объекта": object_data["object_uid"],
                    "Название объекта": object_data["title"],
                    "Время работы": strftime("%H:%M:%S", gmtime(work_time)),
                }
            )

    dataframe_ = pd.DataFrame(objects_info)
    return dataframe_
    #st.table(dataframe_)