# what is find-contours for
 the aim is to find contours in an image with more explicitness. 
 
 the author gets upset about the function findContours() in OpenCV, so he decides to find an alternative.

the most used algorithms are Sobel, Laplacian and Canny.

I wonder if there exists a better approach.

A candidate seems more sensitive to edges than Sobel, which is to get the variance within the kernel instead of multiply.

the result will be shown in demo, we'll see.