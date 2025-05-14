# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import json
import logging
import os
import threading
import time
import uuid
from builtins import anext

from quart import Quart, make_response, request

from vllm.logger import init_logger

# skip aiohttp from isort since it conflicts with yapf
import aiohttp  # isort: skip
'''
A Proxy Server for Disaggregated Inference that has the following features
1. Immediate return of token(s) from prefill server
2. Handles both streaming and non-streaming use cases
3. Forwards requests to both prefill and decode immediately before processing
   prefill output
4. Makes responses from the prefill and decode server appear as if they are
   coming from a single server
'''

# Configure logging
logger = init_logger(__name__)

# Default to WARNING, can be changed to DEBUG for more verbose logging
logger.setLevel(logging.WARNING)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)

workers_lock = threading.RLock()
prefill_workers = []  # [(ip, api_server_port)]
decode_workers = []  # [(ip, api_server_port)]
p_selector = 0
d_selector = 0


async def check_health(ip, port, timeout=5):
    url = f"http://{ip}:{port}/health"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    logger.debug("Service <%s:%s> is healthy", ip, port)
                    return True
                else:
                    logger.warning(
                        "Service <%s:%s> is not healthy. Status code: %s", ip,
                        port, response.status)
                    return False
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning("Error checking health on Service <%s:%s>: %s", ip,
                           port, e)
            return False


def refresh_worker_status(etcd_addr):
    import asyncio

    import etcd3
    host, port = etcd_addr.split(":")
    etcd = etcd3.client(host=host, port=port)

    async def check_workers_health(workers_to_check):
        """Check health of multiple workers concurrently"""
        tasks = []
        for ip, port in workers_to_check:
            tasks.append(check_health(ip, port))
        results = await asyncio.gather(*tasks)
        return [
            worker for worker, is_healthy in zip(workers_to_check, results)
            if is_healthy
        ]

    def run_async_health_checks(workers_to_check):
        """Run async health checks in the current thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        healthy_workers = loop.run_until_complete(
            check_workers_health(workers_to_check))
        loop.close()
        return healthy_workers

    while True:
        new_prefill_workers = []
        new_decode_workers = []

        # Collect workers from etcd
        for value, meta in etcd.get_prefix("/workers/"):
            role, kv_ip, api_port = meta.key.decode().replace("/workers/",
                                                              "").split("/")
            if role == "prefill":
                new_prefill_workers.append((kv_ip, api_port))
            else:
                assert role == "decode"
                new_decode_workers.append((kv_ip, api_port))

        logger.debug(
            "registered prefill workers: %s \n healthy decode workers: %s",
            new_prefill_workers, new_decode_workers)

        # Check health of all workers
        healthy_prefill_workers = run_async_health_checks(new_prefill_workers)
        healthy_decode_workers = run_async_health_checks(new_decode_workers)

        # Update workers list with only healthy workers
        with workers_lock:
            prefill_workers.clear()
            prefill_workers.extend(healthy_prefill_workers)
            decode_workers.clear()
            decode_workers.extend(healthy_decode_workers)

        logger.debug(
            "healthy prefill workers: %s \n healthy decode workers: %s",
            healthy_prefill_workers, healthy_decode_workers)

        time.sleep(3)


async def forward_request(url, data, request_id, request_type="unknown"):
    logger.debug("Starting %s request to %s", request_type, url)
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            headers = {
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
                "X-Request-Id": f"{request_id}",
            }
            logger.debug("%s request attempting connection", request_type)
            try:
                async with session.post(url=url, json=data,
                                        headers=headers) as response:
                    logger.debug("%s request connected", request_type)
                    yield "connection acquired"
                    if response.status == 200:
                        # Stream the response
                        async for chunk in response.content:
                            if chunk:  # ensure chunk isn't empty
                                if isinstance(chunk, bytes):
                                    chunk = chunk.decode('utf-8')
                                yield chunk
                    else:
                        error_text = await response.text()
                        logger.error("Error from %s (%s): %s", url,
                                     request_type, error_text)
                        yield json.dumps({
                            "error":
                            f"Request failed with status {response.status}",
                            "details": error_text
                        }) + '\n'
            except Exception as e:
                logger.error("Request failed for %s: %s", request_type, str(e))
                yield json.dumps({"error": str(e)}) + '\n'
    except Exception as e:
        logger.error("Session creation failed for %s: %s", request_type,
                     str(e))
        yield json.dumps({"error": str(e)}) + '\n'


async def handle_prefill_response(prefill_response, streaming, endpoint,
                                  request_id, request_time):
    # Handle prefill request
    # prefill_url = f"http://{app.args.prefill_ip}:{app.args.prefill_port}{endpoint}"
    logger.debug("Request at %s: Starting prefill", request_time)
    async for chunk in prefill_response:
        if chunk and streaming:
            if "[DONE]" in chunk:
                continue
            chunk = replace_key_in_data_string(chunk, "id", request_id)
            chunk = replace_key_in_data_string(chunk, "finish_reason", None)
            logger.debug("prefill chunk %s", chunk)
            yield chunk
        else:
            pass


async def handle_decode_response(decode_response, streaming, endpoint,
                                 request_id, request_time):
    # Handle decode request
    # decode_url = f"http://{app.args.decode_ip}:{app.args.decode_port}{endpoint}"
    logger.debug("Request at %s: Starting decode", request_time)
    seen_indices = set()
    async for chunk in decode_response:
        if chunk and streaming:
            index = read_key_from_data_string(chunk, "index")
            logger.debug("index: %s, seen_indices: %s", index, seen_indices)
            # Logic to not stream first token from prefill again.
            if index in seen_indices:
                chunk = replace_key_in_data_string(chunk, "id", request_id)
                logger.debug("decode chunk %s", chunk)
                yield chunk
            else:
                seen_indices.add(index)
                logger.debug("skipping decode chunk %s", chunk)
        elif chunk:
            yield chunk


@app.route('/v1/completions', methods=['POST'])
@app.route('/v1/chat/completions', methods=['POST'])
async def handle_request():
    global p_selector, d_selector, workers_lock
    endpoint = request.path
    request_time = str(time.time())
    logger.debug("Processing request at %s", request_time)
    original_request_data = await request.get_json()
    # Logic is different based on if we are streaming the responses
    streaming = original_request_data.get('stream', False)
    prefill_request = original_request_data.copy()
    # Prefill server returns only the first token
    prefill_request['max_tokens'] = 1

    uid = f"{uuid.uuid4()}"
    if app.args.etcd:
        # dynamic mode
        logger.info("running proxy for dynamic xPyD " \
            "with ectd addr %s", app.args.etcd)
        # 1. wait for workers to come alive to make a P/D pair
        # TODO: add fallback support when there is only decode workers
        ready = False
        while not ready:
            with workers_lock:
                if len(prefill_workers) == 0 or len(decode_workers) == 0:
                    logger.info(
                        "No available prefill workers or decode workers,"
                        "sleep and wait for 3s...")
                else:
                    ready = True
            if not ready:
                time.sleep(3)

        # 2. round robin select prefill and decode server
        with workers_lock:
            p_selector %= len(prefill_workers)
            d_selector %= len(decode_workers)

            prefill_ip, prefill_port = prefill_workers[p_selector]
            decode_ip, decode_port = decode_workers[d_selector]

        prefill_request_id = f"cmpl-{uid}_{decode_ip}:{decode_port}"
        decode_request_id = f"cmpl-{uid}_{prefill_ip}:{prefill_port}"

        p_selector += 1
        d_selector += 1
    else:
        # static 1p1d mode
        prefill_ip = app.args.prefill_ip
        prefill_port = app.args.prefill_port
        decode_ip = app.args.decode_ip
        decode_port = app.args.decode_port
        prefill_request_id = f"cmpl-{uid}"
        decode_request_id = f"cmpl-{uid}"

    async def streaming_responses(original_request_data, prefill_request):
        try:
            prefill_url = f"http://{prefill_ip}:{prefill_port}{endpoint}"
            decode_url = f"http://{decode_ip}:{decode_port}{endpoint}"

            logger.info("Routing prefill request %s to %s", prefill_request_id,
                        prefill_url)
            prefill_response = forward_request(prefill_url, prefill_request,
                                               prefill_request_id, "prefill")
            logger.info("Routing decode request %s to %s", decode_request_id,
                        decode_url)
            decode_response = forward_request(decode_url,
                                              original_request_data,
                                              decode_request_id, "decode")

            prefill_task = asyncio.create_task(anext(prefill_response))
            decode_task = asyncio.create_task(anext(decode_response))

            await prefill_task
            async for chunk in handle_prefill_response(prefill_response,
                                                       streaming, endpoint,
                                                       uid, request_time):
                yield chunk

            await decode_task
            async for chunk in handle_decode_response(decode_response,
                                                      streaming, endpoint, uid,
                                                      request_time):
                yield chunk

        except Exception as e:
            logger.exception("Error in request at %s", request_time)
            yield json.dumps({
                "error": str(e),
                "timestamp": request_time
            }) + '\n'

    response = await make_response(
        streaming_responses(original_request_data, prefill_request), {
            'Content-Type': 'application/x-ndjson',
            'Transfer-Encoding': 'chunked'
        })
    response.timeout = None
    return response


def replace_key_in_data_string(data_string, key_to_replace, new_value):
    logger.debug("Received %s in replace_key_in_data_string", data_string)

    # Preserve trailing newline if present
    trailing_newline = '\n' if data_string.endswith('\n') else ''

    # Remove trailing whitespace (including newline) for processing
    data_string = data_string.rstrip()

    # Check if the string starts with "data: "
    if data_string.startswith("data: "):
        # Remove the "data: " prefix
        json_string = data_string[6:]
        prefix = "data: "
    else:
        json_string = data_string
        prefix = ""

    def replace_nested(obj, key, value):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == key:
                    obj[k] = value
                else:
                    replace_nested(v, key, value)
        elif isinstance(obj, list):
            for item in obj:
                replace_nested(item, key, value)

    try:
        # Parse the JSON string
        data = json.loads(json_string)

        # Replace the specified key with the new value
        replace_nested(data, key_to_replace, new_value)

        # Convert back to JSON string without spaces after colons
        modified_json = json.dumps(data, separators=(',', ':'))

        # Return with original prefix and preserved trailing newline
        return f"{prefix}{modified_json}{trailing_newline}"
    except json.JSONDecodeError as e:
        logger.debug(
            "JSON decode error: %s, maybe because there is no JSON "
            "in this string %s", e, data_string)
        return data_string  # Return original string if JSON parsing fails


def read_key_from_data_string(data_string, key_to_read):
    logger.debug("Received %s in read_key_from_data_string", data_string)

    # Remove trailing whitespace (including newline) for processing
    data_string = data_string.strip()

    # Check if the string starts with "data: "
    if data_string.startswith("data: "):
        # Remove the "data: " prefix
        json_string = data_string[6:]
    else:
        json_string = data_string

    def find_nested(obj, key):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == key:
                    return v
                elif isinstance(v, (dict, list)):
                    result = find_nested(v, key)
                    if result is not None:
                        return result
        elif isinstance(obj, list):
            for item in obj:
                result = find_nested(item, key)
                if result is not None:
                    return result
        return None

    try:
        # Parse the JSON string
        data = json.loads(json_string)

        # Find and return the value for the specified key
        result = find_nested(data, key_to_read)

        return result
    except json.JSONDecodeError as e:
        logger.debug(
            "JSON decode error: %s, maybe because there is no JSON "
            "in this string %s", e, json_string)
        return None  # Return None if JSON parsing fails


def enable_debug_logging():
    """Call this function to enable debug logging"""
    logger.setLevel(logging.DEBUG)


if __name__ == '__main__':
    # Uncomment the next line to enable debug logging
    enable_debug_logging()
    parser = argparse.ArgumentParser(
        description=
        'Proxy server for prefill and decode servers that immediately returns '
        'the first token.')
    parser.add_argument(
        '--prefill-ip',
        default='localhost',
        help='IP address for prefill server (default: localhost)')
    parser.add_argument(
        '--decode-ip',
        default='localhost',
        help='IP address for decode server (default: localhost)')
    parser.add_argument('--etcd',
                        default=None,
                        help='etcd host ip:port to enable dynamic routing')
    parser.add_argument('--prefill-port',
                        type=int,
                        default=8100,
                        help='Port for prefill server (default: 8100)')
    parser.add_argument('--decode-port',
                        type=int,
                        default=8200,
                        help='Port for decode server (default: 8200)')

    args = parser.parse_args()
    app.args = args
    if args.etcd:
        threading.Thread(target=refresh_worker_status,
                         args=(args.etcd, ),
                         daemon=True).start()
    app.run(port=8000)
