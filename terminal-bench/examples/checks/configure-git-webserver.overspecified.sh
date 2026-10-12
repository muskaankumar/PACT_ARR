#!/bin/bash
# check_overspecified.sh
#!/bin/bash

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y expect curl

mkdir -p ~/.ssh
echo "Host localhost
    StrictHostKeyChecking no
    UserKnownHostsFile=/dev/null" > ~/.ssh/config
chmod 600 ~/.ssh/config

cat > git_test.exp << 'EOF'
#!/usr/bin/expect -f
set timeout 10

spawn git clone git@localhost:/git/server server-test
expect {
    "yes/no" {
        send "yes\r"
        exp_continue
    }
    "password:" {
        send "password\r"
    }
}
expect eof

spawn bash -c "cd server-test && echo 'hello world' > hello.html && git add hello.html && git config --local user.email 'test@example.com' && git config --local user.name 'Test User' && git commit -m 'Add hello.html'"
expect eof

spawn bash -c "cd server-test && git push origin master"
expect "password:"
send "password\r"
expect eof

exit
EOF

chmod +x git_test.exp
./git_test.exp

sleep 2

HTTP_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/hello.html)

if [ "$HTTP_RESPONSE" = "200" ]; then
    CONTENT=$(curl -s http://localhost:8080/hello.html)
    if [ "$CONTENT" = "hello world" ]; then
        echo "TEST PASSED: Content matches expected output"
    else
        echo "TEST FAILED: Content does not match expected output"
        echo "Expected: 'hello world'"
        echo "Got: '$CONTENT'"
        exit 1
    fi
else
    echo "TEST FAILED: Web server returned HTTP $HTTP_RESPONSE"
    exit 1
fi

RAW_BYTES=$(curl -s http://localhost:8080/hello.html | wc -c)
if [ "$RAW_BYTES" -ne 12 ]; then
    echo "TEST FAILED: hello.html response body is $RAW_BYTES bytes, expected exactly 12 bytes (hello world + newline)"
    exit 1
fi

HOOK_PERMS=$(stat -c "%a" /git/server/hooks/post-receive 2>/dev/null)
if [ "$HOOK_PERMS" != "755" ]; then
    echo "TEST FAILED: post-receive hook permissions are $HOOK_PERMS, expected 755"
    exit 1
fi

echo "ALL CHECKS PASSED"