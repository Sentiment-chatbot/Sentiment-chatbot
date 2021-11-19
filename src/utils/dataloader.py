import pandas as pd
import json

def load(
    data_path,
    ignored_columns=[
        "profile.persona-id",
        "profile.persona.persona-id",
        "profile.persona.human",
        "profile.persona.computer",
        "profile.emotion.emotion-id",
        "profile.emotion.situation",
        "talk.id.profile-id",
        "talk.id.talk-id",
    ],
):
    """ Load dataset from raw json file (+ simple pre-processing) """

    data = None
    with open(data_path, "r", encoding="UTF8") as f:
        data = json.loads(f.read())

    # Load
    df = pd.json_normalize(data)
    df = df.drop(ignored_columns, axis=1)

    # Rename columns
    columns = dict()
    for column in df.columns:
        columns.update(dict({column: column.split(".")[-1]}))
    df = df.rename(columns=columns)

    # Get details
    df.rename(columns={"type": "emotion"}, inplace=True)
    df["emotion"] = df["emotion"].replace(
        {f"E{i+10}": label for i, label in enumerate(label_list)}
    )

    return df
