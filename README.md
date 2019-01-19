# ISP2019 Docker

<ol> 
<li> Run "docker build -t image_name ." </li>
<li> Run "docker run -v "$(pwd)/results:/example/results" image_name" - this will mount conrainer's  /example/results directory to ./results directory on your host machine</li>
</ol>