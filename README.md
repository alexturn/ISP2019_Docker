# ISP2019 Docker

<ol> 
<li> Run "docker build -t image_name ." </li>
<li> Run "docker run -v "$(pwd)/results:/calibration/results" image_name" - this will mount conrainer's  calibration/results directory to ./results directory on your host machine</li>
</ol>