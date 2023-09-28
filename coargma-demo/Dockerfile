FROM node:18.17.1
COPY . /app_frontend
WORKDIR /app_frontend
RUN apt-get update && apt-get install make
RUN make .
CMD ["npm", "install"]
CMD ["npm", "run", "dev"]