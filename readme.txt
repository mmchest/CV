用python实现图片的读取与显示、裁切、颜色变换、旋转和投射。
先定义函数，再通过函数实现。

第一个函数crop实现裁切操作。输出图片和要截到的像素位置，返回裁切后的图片。
第二个函数colorshift实现颜色变换。输入图像和RGB三个颜色要改变的值（亮度），返回颜色变换后的图片
第三个函数rotate实现图像的旋转。默认旋转中心是图片的中心，输入图像和旋转的角度，返回旋转一定角度后的图像。
第四个函数perspective实现图像的投射。这个函数只实现左下-右上对角方向的对称压缩。输入图片和要压缩的像素值，返回投射后的图片。