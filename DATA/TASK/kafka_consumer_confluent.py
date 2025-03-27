import json
from confluent_kafka import Consumer, KafkaError

KAFKA_BROKER = 'pkc-921==========개인정보-----------'
KAFKA_TOPIC = 'input_data'
KAFKA_GROUP_ID = 'input_data_consumer_group'
KAFKA_CONFIG = {
    'bootstrap.servers': KAFKA_BROKER,
    'group.id': KAFKA_GROUP_ID,
    'auto.offset.reset': 'earliest',
    'sasl.mechanisms': 'PLAIN',
    'security.protocol': 'SASL_SSL',
    'sasl.username': 'O6QYLINQLEIFOXND',
    'sasl.password': '4LIf9WkBmOywbAPhChHL1BiVZR9yV0t90Rd7oqIwCrOe7a52pd9S/5y5DEyXy9+U',
}


def consume_data():
    consumer = Consumer(KAFKA_CONFIG)
    consumer.subscribe([KAFKA_TOPIC])
    data_list = []

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f"Kafka 에러: {msg.error()}")
                    break

            try:
                data = json.loads(msg.value().decode('utf-8'))
                data_list.append(data)
            except json.decoder.JSONDecodeError as e:
                print(f"JSON 형태 오류: {e} - 메시지: {msg.value().decode('utf-8')}")

    except KeyboardInterrupt:
        print("종료 요청 받음. Kafka Consumer 종료 중...")

    finally:
        consumer.close()
        print("Kafka Consumer가 종료되었습니다.")

    return data_list
