FROM python:3.7

MAINTAINER alexmtcore@gmail.com
WORKDIR calibration

RUN apt-get update && apt-get install -y texlive

RUN pip install numpy scipy matplotlib tqdm

ADD code ./code
ADD latex ./latex
ADD run.sh ./

RUN chmod +x run.sh
VOLUME /calibration/results

CMD ./run.sh