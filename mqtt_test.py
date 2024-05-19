import paho.mqtt.client as mqtt
import pexpect
import json 

# Function to load configuration from JSON file
def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def run_command_with_password(command, password):
    # Join the command list into a single string
    command_str = ' '.join(['sudo'] + command)
    
    # Spawn the command
    child = pexpect.spawn(command_str)
    
    # Expect the password prompt
    child.expect('password for')
    
    # Send the password followed by a newline character
    child.sendline(password)
    
    # Wait for the command to complete
    child.expect(pexpect.EOF)
    
    # Print the output of the command
    print(child.before.decode())

def start_mqtt() : 
    
    # Enable Mosquitto to start on boot
    run_command_with_password(['systemctl', 'enable', 'mosquitto'], sudo_password)

    # Start the Mosquitto service
    run_command_with_password(['systemctl', 'start', 'mosquitto'], sudo_password)


################CONFIG#################################

# Load configuration
config = load_config('config.json')

# Define the MQTT broker settings
broker = "localhost"  
port = 1883  # Default MQTT port

# Define the topic and the message
topic = "test"  # Replace with your topic
message = "Hello, Node-RED!"  # Replace with your message

# Define your password here (note: this is insecure and not recommended for production use)
sudo_password = config['sudo_pass'] #just for testing 

#######################################################

def mqtt_connect(topic,message) : 
    # Create an MQTT client instance
    client = mqtt.Client()

    # Connect to the MQTT broker
    client.connect(broker, port, 60)

    # Publish the message to the specified topic
    client.publish(topic, message)

    # Disconnect from the broker
    client.disconnect()

    print(f"Message '{message}' sent to topic '{topic}'")




