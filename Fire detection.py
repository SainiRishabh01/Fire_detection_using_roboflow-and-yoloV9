from roboflow import Roboflow
rf=Roboflow(api_key="K3iF6N8dhQB937bXevqZ")

project=rf.workspace().project("fire-us9wz")
model=project.version(1).model


print(model.predict("a.jpg",confidence=40, overlap=30).json())

model.predict("a.jpg", confidence=40, overlap=30).save("b.jpg")


