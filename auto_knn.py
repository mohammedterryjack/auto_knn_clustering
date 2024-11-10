from typing import Generator

from numpy import ndarray, array, diff, argmax, ones_like, where, uint8, zeros
from numpy import any as np_any
from cv2 import (
    resize,
    cvtColor,
    threshold,
    findContours,
    INTER_AREA,
    INTER_NEAREST,
    RETR_EXTERNAL,
    CHAIN_APPROX_SIMPLE,
    COLOR_BGR2GRAY
)

from sklearn.cluster import KMeans


 

def index_elbow_point(inertia:list[int]) -> int:
    second_derivative = diff(diff(inertia))
    return argmax(second_derivative) + 1

def select_optimal_model(trained_models:list[KMeans]) -> KMeans:
    return trained_models[
        index_elbow_point(
            inertia=[model.inertia_ for model in trained_models]
        )
    ]

def select_lightest_cluster_index(pixel_values:ndarray, labels:ndarray, max_cluster_id:int) -> int:
    intensities = [
        pixel_values[labels == i].mean()        
        for i in range(max_cluster_id)
    ]
    lightest_intensity = max(intensities)
    return intensities.index(lightest_intensity)

def cluster_image_by_intensity(
    grayscale_image:ndarray,
    white_normalised_threshold:float = 0.5
) -> tuple[ndarray,ndarray]:    
    """Returns image mask of cluster ids
    and a second mask of just the cluster id with the highest mean intensity
    """
    image_normalised = grayscale_image / 255
    pixels = image_normalised.reshape(-1,1)
    pixel_labels = -1*ones_like(pixels)
    indexes_coloured_pixels = where(pixels>white_normalised_threshold)
    coloured_pixels = pixels[indexes_coloured_pixels].reshape(-1,1)

    if not np_any(coloured_pixels):
        return pixel_labels.reshape(image_normalised.shape),zeros(image_normalised.shape)

    models = [KMeans(n_clusters=k) for k in range(1,5)]
    [model.fit(coloured_pixels) for model in models]
    model = select_optimal_model(trained_models=models)
    coloured_pixel_labels = model.predict(coloured_pixels)
    pixel_labels[indexes_coloured_pixels]= coloured_pixel_labels
    image_clustered = pixel_labels.reshape(image_normalised.shape)
    i = select_lightest_cluster_index(
        pixel_values=pixels,
        labels=pixel_labels,
        max_cluster_id=model.n_clusters
    )
    return image_clustered, where(image_clustered == i, 1, 0)

 

def convert_grayscale_image_to_black_white_mask(
    grayscale_image:ndarray,
    thumbnail_size:tuple[int,int]=(256,256),
    white_pixel_threshold:int=150
) -> ndarray:
    height,width = grayscale_image.shape
    grayscale_thumbnail = resize(grayscale_image, thumbnail_size, interpolation=INTER_AREA)
    grayscale_image_low_res = resize(grayscale_thumbnail, (width, height), interpolation=INTER_NEAREST)
    _,mask = cluster_image_by_intensity(grayscale_image=grayscale_image_low_res)
    return mask.astype(uint8)


height,width,_ = intensity_image.shape
mask = convert_grayscale_image_to_black_white_mask(
    grayscale_image=cvtColor(intensity_image, COLOR_BGR2GRAY),
    thumbnail_size=(256,256),
    white_pixel_threshold=150
)
contours, _ = findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)


 
