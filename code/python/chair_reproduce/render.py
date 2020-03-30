import sys
import numpy as np
import trimesh
import os

def render(path, name):
    obj_name = os.path.join(path, name)
    # print logged messages
    trimesh.util.attach_to_log()

    # load a mesh
    mesh = trimesh.load(obj_name)

    # get a scene object containing the mesh, this is equivalent to:
    # scene = trimesh.scene.Scene(mesh)
    scene = mesh.scene()

    # add a base transform for the camera which just
    # centers the mesh into the FOV of the camera
    scene.set_camera()

    # a 45 degree homogenous rotation matrix around
    # the Y axis at the scene centroid
    rotate = trimesh.transformations.rotation_matrix(
        angle=np.radians(180.0),
        direction=[0, 1, 0],
        point=scene.centroid)

    for i in range(1):
        trimesh.constants.log.info('Saving image %d', i)

        # rotate the camera view transform
        camera_old, _geometry = scene.graph['camera']
        camera_new = np.dot(camera_old, rotate)

        # apply the new transform
        scene.graph['camera'] = camera_new

        # saving an image requires an opengl context, so if -nw
        # is passed don't save the image
        try:
            # increment the file name
            file_name = os.path.join(path, name.split('.')[0] + '.png')
            # save a render of the object as a png
            png = scene.save_image(resolution=[1920, 1080], visible=True)
            with open(file_name, 'wb') as f:
                f.write(png)
                f.close()

        except BaseException as E:
            print("unable to save image", str(E))

if __name__ == '__main__':
    path = sys.argv[1]
    filelist = [file for file in os.listdir(path) if file.endswith('.obj')]
    for file in filelist:
        render(path, file)
