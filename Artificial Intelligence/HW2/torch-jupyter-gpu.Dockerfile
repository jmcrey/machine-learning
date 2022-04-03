FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

# Install Jupyterlab
RUN conda install jupyterlab sacrebleu -c conda-forge

# Set workdir and expose port
WORKDIR /src
EXPOSE 8888

# Jupyter stuff
# ENV JUPYTER_RUNTIME_DIR="/src/.runtime/"
# ENV JUPYTER_CONFIG_DIR="/src/.config/"
# ENV JUPYTER_CONFIG_PATH="/src/.config/"
# ENV JUPYTER_PATH="/src/"
# ENV JUPYTER_DATA_DIR="/src/"

# Start jupyter notebook
CMD [ "jupyter", "notebook", "--no-browser", "--ip=0.0.0.0", "--allow-root" ]
