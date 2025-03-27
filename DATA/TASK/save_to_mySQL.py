import mysql.connector

mysql_conn = mysql.connector.connect(
    host="localhost", user="root", password="password", database="project_db"
)
mysql_cursor = mysql_conn.cursor()

def save_to_mysql(results):
    query = """
        INSERT INTO analysis_results (name, description, amount, predicted_risk, analyzed_at)
        VALUES (%s, %s, %s, %s, %s)
    """
    for result in results:
        values = (
            result['name'],
            result['description'],
            result['amount'],
            result['predicted_risk'],
            result['analyzed_at']
        )
        mysql_cursor.execute(query, values)
    mysql_conn.commit()
    print("AI Analysis results saved to MySQL.")
