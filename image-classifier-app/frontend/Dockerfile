FROM node:18-alpine

WORKDIR /app

COPY image-classifier-web/package*.json ./
RUN npm install

COPY image-classifier-web/ .

RUN npm run build

CMD ["npm", "start"]
