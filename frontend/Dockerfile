FROM node:18-slim
WORKDIR /app
# Copy dependency list first to use Docker's cache
COPY package.json .
RUN npm install
COPY . .
# React standard port
EXPOSE 3000
CMD ["npm", "start"]