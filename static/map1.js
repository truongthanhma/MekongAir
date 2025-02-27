// Tạo bản đồ với Leaflet.js
var map = L.map('map', {
    center: [10.1, 105.5],  // Tọa độ trung tâm Đồng bằng Sông Cửu Long
    zoom: 9,
    scrollWheelZoom: false  // Vô hiệu hóa cuộn chuột mặc định để phóng to/thu nhỏ
});

// Thêm tile layer từ OpenStreetMap
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18,
}).addTo(map);

// #############################

// Hàm tính khoảng cách giữa hai điểm (lat, lon) bằng Haversine
function calculateDistance(lat1, lon1, lat2, lon2) {
    const R = 6371; // Bán kính Trái Đất tính bằng km
    const dLat = (lat2 - lat1) * (Math.PI / 180);
    const dLon = (lon2 - lon1) * (Math.PI / 180);
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
              Math.cos(lat1 * (Math.PI / 180)) * Math.cos(lat2 * (Math.PI / 180)) *
              Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
}

// Hàm tìm huyện gần nhất từ vị trí hiện tại
function suggestNearestDistrict(lat, lon) {
    let nearestDistrict = null;
    let minDistance = Infinity;

    for (const district in districtCoords) {
        const { lat: districtLat, lon: districtLon } = districtCoords[district];
        const distance = calculateDistance(lat, lon, districtLat, districtLon);
        if (distance < minDistance) {
            minDistance = distance;
            nearestDistrict = district;
        }
    }

    if (nearestDistrict) {
        alert(`Gợi ý huyện gần nhất: ${nearestDistrict}`);
        searchDistrict(nearestDistrict);
    } else {
        alert("Không tìm thấy huyện gần nhất.");
    }
}

// Xử lý hiển thị thông báo khi người dùng nhấn vào ô tìm kiếm
document.getElementById('search-district').addEventListener('focus', function() {
    // Kiểm tra sessionStorage xem prompt đã hiển thị chưa
    if (!sessionStorage.getItem('locationPromptShown')) {
        document.getElementById('location-prompt').style.display = 'block';
        sessionStorage.setItem('locationPromptShown', 'true'); // Đánh dấu là đã hiển thị
    }
});

document.getElementById('skip-location').addEventListener('click', function() {
    document.getElementById('location-prompt').style.display = 'none'; // Ẩn hộp thông báo
});

document.getElementById('use-location').addEventListener('click', function() {
    document.getElementById('location-prompt').style.display = 'none';

    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition((position) => {
            const latitude = position.coords.latitude;
            const longitude = position.coords.longitude;
            suggestNearestDistrict(latitude, longitude);
        }, (error) => {
            console.error("Error fetching location:", error);
            alert("Không thể lấy vị trí hiện tại. Vui lòng tìm kiếm thủ công.");
        });
    } else {
        alert("Trình duyệt của bạn không hỗ trợ chức năng định vị.");
    }
});

// #############################
map.getContainer().addEventListener('wheel', function(event) {
    // Ngăn chặn việc zoom toàn bộ trang khi nhấn Ctrl
    event.preventDefault();

    if (!event.ctrlKey) {
        // Kiểm tra và xóa thông báo cũ nếu tồn tại
        var existingNotification = document.getElementById('zoom-notification');
        if (existingNotification) {
            existingNotification.remove();
        }

        // Tạo thông báo mới
        var notification = document.createElement('div');
        notification.id = 'zoom-notification';
        notification.style.position = 'fixed';
        notification.style.top = '50%';
        notification.style.left = '50%';
        notification.style.transform = 'translate(-50%, -50%) scale(0.8)';
        notification.style.color = '#fff';  
        notification.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
        notification.style.padding = '15px 30px';
        notification.style.zIndex = '1000';
        notification.style.fontSize = '16px';
        notification.style.boxShadow = '0 10px 20px rgba(0, 0, 0, 0.2)';
        notification.style.opacity = '0';
        notification.style.transition = 'all 0.5s ease';
        notification.textContent = "Press Ctrl + scroll to zoom in/out";
        document.body.appendChild(notification);

        // Hiệu ứng fade-in và zoom-in
        setTimeout(function() {
            notification.style.opacity = '1';
            notification.style.transform = 'translate(-50%, -50%) scale(1)';
        }, 10);

        // Xóa thông báo sau 3 giây với hiệu ứng fade-out và zoom-out
        setTimeout(function() {
            notification.style.opacity = '0';
            notification.style.transform = 'translate(-50%, -50%) scale(0.8)';
            setTimeout(function() {
                notification.remove();
            }, 500);  // Đợi hiệu ứng fade-out hoàn thành trước khi xóa
        }, 1000);

    } else {
        // Chỉ thực hiện zoom bản đồ khi người dùng nhấn Ctrl và cuộn chuột
        if (event.deltaY < 0) {
            map.zoomIn();  // Phóng to nếu cuộn lên
        } else {
            map.zoomOut(); // Thu nhỏ nếu cuộn xuống
        }
    }
});


// Biến lưu thông tin các huyện
var districtCoords = {};
var districtNames = [];  // Mảng để lưu tên huyện cho autocomplete
var districts = {};

// Hiển thị modal hướng dẫn khi tải trang
document.addEventListener('DOMContentLoaded', function() {
    var guideModal = new bootstrap.Modal(document.getElementById('guide-modal'), {
        backdrop: 'static', 
        keyboard: false     
    });
    guideModal.show();

    // Khi người dùng nhấn nút "Đã hiểu", ẩn modal và lưu trạng thái vào sessionStorage
    document.getElementById('understood-btn').addEventListener('click', function() {
        sessionStorage.setItem('modalShown', 'true'); // Lưu trạng thái vào sessionStorage
        var modal = bootstrap.Modal.getInstance(document.getElementById('guide-modal'));
        modal.hide(); // Ẩn modal
    });

    // Khi người dùng nhấn nút "Close", cũng lưu trạng thái vào sessionStorage
    document.getElementById('close-modal').addEventListener('click', function() {
        sessionStorage.setItem('modalShown', 'true'); // Lưu trạng thái vào sessionStorage
        var modal = bootstrap.Modal.getInstance(document.getElementById('guide-modal'));
        modal.hide(); // Ẩn modal
    });
    
});

// Lấy dữ liệu các huyện từ server
fetch('/get_districts')
    .then(response => response.json())
    .then(data => {
        districts = data;  // Gán dữ liệu lấy từ server vào biến districts

        // Duyệt qua từng tỉnh và huyện để hiển thị marker trên bản đồ
        for (var province in districts) {
            var districtData = districts[province];
            for (var district in districtData) {
                var coords = districtData[district];
                
                // Lưu tọa độ huyện vào biến districtCoords
                districtCoords[district] = { lat: coords.lat, lon: coords.lon };

                // Thêm tên huyện vào mảng gợi ý tìm kiếm
                districtNames.push(district);

                // Thêm marker lên bản đồ
                var marker = L.marker([coords.lat, coords.lon]).addTo(map)
                    .bindPopup(`${district}, ${province}`)
                    .on('click', function(e) {
                        var popupContent = this.getPopup().getContent().split(", ");
                        var districtName = popupContent[0];
                        console.log(`Marker của huyện ${districtName} đã được nhấn`); 
                        fetchAQIPrediction(districtName, e);  // Truyền sự kiện click
                    });
            }
        }

        // Khởi tạo autocomplete cho thanh tìm kiếm
        // $("#search-district").autocomplete({
        //     source: districtNames
        // });
    })
    .catch(error => {
        console.error('Error fetching district data:', error);
    });

    function fetchAQIPrediction(districtName, e) {

        // Kiểm tra và xóa form chọn ngày giờ nếu đã tồn tại
        let existingFormContainer = document.querySelector('.form-container');
        if (existingFormContainer) {
            document.body.removeChild(existingFormContainer);
        }
        
        const formContainer = document.createElement('div');
        formContainer.className = 'form-container';
    
        const formTitle = document.createElement('h3');
        formTitle.innerText = `Select date and time for AQI forecast in ${districtName} district`;
        formContainer.appendChild(formTitle);
    
        // Input ngày
        const inputDate = document.createElement('select');
        inputDate.id = 'input-date';
        formContainer.appendChild(inputDate);
    
        // Tạo phần tử select để người dùng chọn giờ
        const selectHour = document.createElement('select');
        selectHour.id = 'select-hour';
        formContainer.appendChild(selectHour);
    
        // Nút Xác nhận
        const submitButton = document.createElement('button');
        submitButton.innerText = 'Confirm';
        submitButton.className = 'submit-btn';
        formContainer.appendChild(submitButton);
    
        // Nút Close
        const closeButton = document.createElement('button');
        closeButton.innerText = 'Close';
        closeButton.className = 'close-btn';
        formContainer.appendChild(closeButton);
    
        document.body.appendChild(formContainer);
    
        // Gửi yêu cầu lấy danh sách ngày và giờ từ backend
        fetch('/get_available_dates_and_times', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 'district': districtName })
        })
        .then(response => response.json())
        .then(data => {
            if (data.available_dates) {
                const currentDate = new Date();
    
                Object.keys(data.available_dates).forEach(date => {
                    const dateObj = new Date(date);
                    if (dateObj > currentDate.setHours(0,0,0,0)) {  
                        const option = document.createElement('option');
                        option.value = date;
                        option.text = date;
                        inputDate.appendChild(option);
                    }
                });
    
                inputDate.addEventListener('change', () => {
                    const selectedDate = inputDate.value;
                    updateHourOptions(selectedDate, data.available_dates[selectedDate]);
                });
    
                if (inputDate.options.length > 0) {
                    inputDate.value = inputDate.options[0].value;
                    updateHourOptions(inputDate.value, data.available_dates[inputDate.value]);
                }
            }
        })
        .catch(error => {
            console.error('Error fetching available dates and times:', error);
        });
    
        function updateHourOptions(selectedDate, availableHours) {
            const selectHour = document.getElementById('select-hour');
            selectHour.innerHTML = '';  // Xóa các tùy chọn giờ hiện tại
            
            const currentDate = new Date();
            const selectedDateObj = new Date(selectedDate);
        
            availableHours.forEach(hour => {
                const option = document.createElement('option');
                
                // So sánh trực tiếp giờ và phút, không chuyển đổi
                const [hourOnly, minute] = hour.split(':'); 
                const hourInt = parseInt(hourOnly, 10);
                const minuteInt = parseInt(minute, 10);
                
                // Nếu là ngày hiện tại, chỉ hiển thị giờ lớn hơn giờ hiện tại
                if (selectedDateObj.toDateString() === currentDate.toDateString()) {
                    if (hourInt >= currentDate.getHours() || (hourInt >= currentDate.getHours() && minuteInt >= currentDate.getMinutes())) {  
                        option.value = hour;
                        option.text = hour;
                        selectHour.appendChild(option);
                    }
                } else {
                    // Nếu là ngày trong tương lai, hiển thị tất cả giờ từ 0h đến 23h
                    option.value = hour;
                    option.text = hour;
                    selectHour.appendChild(option);
                }
            });
        }
    
        submitButton.addEventListener('click', () => {
            const selectedHour = selectHour.value;
            const selectedDate = inputDate.value;
    
            fetch('/get_aqi', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 'district': districtName, 'hour': selectedHour, 'date': selectedDate })
            })
            .then(response => response.json())
            .then(data => {
                if (data.aqi_prediction) {
                    showInfoTab(districtName, e, data);
                }
            })
            .catch(error => {
                console.error('Error fetching AQI:', error);
            });
    
            document.body.removeChild(formContainer); // Xóa form sau khi chọn
        });
    
        closeButton.addEventListener('click', () => {
            document.body.removeChild(formContainer);
        });
    }
// Theo dõi trạng thái của quá trình huấn luyện
function trackTrainingStatus() {
    const interval = setInterval(() => {
        fetch('/check_training_status')
            .then(response => response.json())
            .then(data => {
                console.log("Training status: ", data.training_in_progress);  // Thêm log để kiểm tra trạng thái
                if (!data.training_in_progress) {
                    // Khi quá trình huấn luyện kết thúc, dừng theo dõi và ẩn thanh progress bar
                    clearInterval(interval);
                    hideProgressBar();
                    alert('Data has been updated. Please try fetching the AQI again.');
                }
            })
            .catch(error => {
                console.log('Error checking training status:', error);
                clearInterval(interval);
                hideProgressBar();
                alert('An error occurred during the update process.');
            });
    }, 1000);  // Kiểm tra trạng thái mỗi giây
}

// kéo thả info-tab
function makeDraggable(element) {
    let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;

    element.onmousedown = dragMouseDown;

    function dragMouseDown(e) {
        e = e || window.event;
        e.preventDefault();

        // Thay đổi con trỏ chuột thành hình bàn tay khi bắt đầu kéo
        document.body.style.cursor = 'grabbing';

        // Lấy vị trí con trỏ chuột lúc bắt đầu
        pos3 = e.clientX;
        pos4 = e.clientY;

        document.onmouseup = closeDragElement;
        document.onmousemove = elementDrag;
    }

    function elementDrag(e) {
        e = e || window.event;
        e.preventDefault();

        // Tính toán vị trí mới của con trỏ chuột
        pos1 = pos3 - e.clientX;
        pos2 = pos4 - e.clientY;
        pos3 = e.clientX;
        pos4 = e.clientY;

        // Đặt vị trí mới của phần tử
        element.style.top = (element.offsetTop - pos2) + "px";
        element.style.left = (element.offsetLeft - pos1) + "px";
    }

    function closeDragElement() {
        // Đặt lại con trỏ chuột về trạng thái mặc định khi ngừng kéo
        document.body.style.cursor = 'default';

        document.onmouseup = null;
        document.onmousemove = null;
    }
}

// Hiển thị info-tab khi có dữ liệu mới
function showInfoTab(districtName, e, data) {
    let infoTab = document.getElementById('info-tab');
    infoTab.classList.add('active');
    document.querySelector('.custom-aqi-title').innerText = `Predict AQI for ${districtName}`;

    // Xóa nội dung cũ trước khi thêm mới
    let tableBody = document.getElementById('aqi-table-body');
    tableBody.innerHTML = '';

    let formattedDate = `${data.date} ${data.hour}`;
    let aqiLevel = getAQILevel(data.aqi_prediction);
    let aqiColorClass = getAQIColorClass(data.aqi_prediction);

    let row = `<tr class="${aqiColorClass}" data-aos="fade-up">
        <td>${formattedDate}</td>
        <td>${Math.round(data.aqi_prediction)} AQI US</td>
        <td>${aqiLevel}</td>
    </tr>`;
    tableBody.innerHTML += row;

    infoTab.style.display = 'block';
    infoTab.style.top = (e.originalEvent.pageY - 100) + 'px';
    infoTab.style.left = e.originalEvent.pageX + 'px';

    makeDraggable(infoTab);
    AOS.refresh();

    // Thêm sự kiện cho nút "Hiển thị Biểu đồ" với districtName và date
    const showChartButton = document.getElementById('show-chart-btn');
    showChartButton.onclick = function() {
        showAQIChart(districtName, data.date); // gọi hàm showAQIChart với districtName và date
    };

    let recommendations = getHealthRecommendations(data.aqi_prediction);
    let recommendationRow = `<tr><td colspan="3" style="color: black; font-weight: bold;">
                              <strong>Health Recommendation:</strong> ${recommendations}
                            </td></tr>`;
    tableBody.innerHTML += recommendationRow;
}

// Hàm để hiển thị biểu đồ AQI trong modal
function showAQIChart(districtName, selectedDate) {
    fetch('/get_daily_aqi', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 'district': districtName, 'date': selectedDate })
    })
    .then(response => response.json())
    .then(dailyData => {
        if (dailyData.hourly_aqi) {
            renderAQIChart(dailyData.hourly_aqi);
            document.getElementById('chart-modal').style.display = 'block'; // Hiển thị cửa sổ biểu đồ
        }
    })
    .catch(error => {
        console.error('Error fetching daily AQI data:', error);
    });
}

// Đóng cửa sổ biểu đồ khi nhấn nút close
document.getElementById('close-chart-modal').addEventListener('click', function() {
    document.getElementById('chart-modal').style.display = 'none';
});


// Hàm vẽ biểu đồ
function renderAQIChart(hourlyAQI) {
    const ctx = document.getElementById('aqiChart').getContext('2d');
    if (window.aqiChartInstance) {
        window.aqiChartInstance.destroy();
    }

    const labels = Object.keys(hourlyAQI);
    const values = Object.values(hourlyAQI);

    window.aqiChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'AQI per Hour',
                data: values,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'AQI Level'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Hours'
                    }
                }
            }
        }
    });
}


// Hiển thị overlay
function showProgressBar() {
    document.getElementById('loading-overlay').style.display = 'flex';
}

// Ẩn overlay
function hideProgressBar() {
    document.getElementById('loading-overlay').style.display = 'none';
}

// Hàm trả về mức độ ô nhiễm dựa trên giá trị AQI
function getAQILevel(aqi) {
    if (aqi <= 50) {
        return 'Good';
    } else if (aqi <= 100) {
        return 'Moderate';
    } else if (aqi <= 150) {
        return 'Unhealthy for Sensitive Groups';
    } else if (aqi <= 200) {
        return 'Unhealthy';
    } else if (aqi <= 300) {
        return 'Very Unhealthy';
    } else {
        return 'Hazardous';
    }
}

// Hàm trả về lớp màu tương ứng với giá trị AQI
function getAQIColorClass(aqi) {
    if (aqi <= 50) {
        return 'aqi-good';
    } else if (aqi <= 100) {
        return 'aqi-moderate';
    } else if (aqi <= 150) {
        return 'aqi-unhealthy';
    } else if (aqi <= 200) {
        return 'aqi-very-unhealthy';
    } else if (aqi <= 300) {
        return 'aqi-hazardous';
    } else {
        return 'aqi-danger';
    }
}

function removeAccents(str) {
    return str.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
}

// Xử lý form tìm kiếm và zoom vào huyện với hiệu ứng chuyển động
function searchDistrict(districtName) {
    // Chuyển đổi tên huyện được nhập thành chữ thường và bỏ dấu để so sánh
    var nameLower = removeAccents(districtName.toLowerCase());

    // Tìm kiếm tên huyện trong districtCoords bằng cách không phân biệt chữ hoa, chữ thường và không dấu
    var foundDistrict = Object.keys(districtCoords).find(function(key) {
        return removeAccents(key.toLowerCase()) === nameLower;
    });

    if (foundDistrict) {
        // Lấy tọa độ từ huyện đã tìm thấy (với tên gốc, không đổi thành chữ thường)
        var coords = districtCoords[foundDistrict];
        map.flyTo([coords.lat, coords.lon], 12, {
            animate: true,
            duration: 1.5
        });
        // fetchAQIPrediction(foundDistrict);
    } else {
        alert("Không tìm thấy huyện " + districtName);
    }
}



// Xử lý gợi ý khi người dùng nhập
document.querySelector("input[type='search']").addEventListener("input", function(event) {
    var searchSuggestions = document.getElementById("search-suggestions");
    var searchValue = removeAccents(event.target.value.toLowerCase());  // Chuyển giá trị tìm kiếm thành chữ thường và bỏ dấu

    if (searchValue === '') {
        // Nếu giá trị tìm kiếm rỗng, ẩn danh sách gợi ý
        searchSuggestions.classList.remove("active");
    } else {
        // Lọc các huyện dựa trên giá trị tìm kiếm đã loại bỏ dấu
        var suggestions = Object.keys(districtCoords).filter(function(district) {
            return removeAccents(district.toLowerCase()).includes(searchValue);  // So sánh không phân biệt hoa/thường và không dấu
        });

        // Hiển thị gợi ý nếu có dữ liệu và giữ nguyên kiểu chữ gốc
        if (suggestions.length > 0) {
            searchSuggestions.innerHTML = '<ul>' + suggestions.map(function(d) {
                return '<li>' + d + '</li>';  // Giữ nguyên kiểu chữ của tên huyện
            }).join('') + '</ul>';
            searchSuggestions.classList.add("active");
        } else {
            searchSuggestions.classList.remove("active");
        }
    }
});



// Xử lý click vào các gợi ý
document.querySelector("#search-suggestions").addEventListener('click', function(event) {
    if (event.target.tagName === 'LI') {
        var selectedDistrict = event.target.innerText;
        document.querySelector("#search-district").value = selectedDistrict;
        searchDistrict(selectedDistrict);  // Gọi hàm tìm kiếm với gợi ý đã chọn
        this.classList.remove("active");  // Ẩn danh sách gợi ý sau khi chọn
    }
});

document.addEventListener('click', function(event) {
    var searchSuggestions = document.getElementById('search-suggestions');
    var searchInput = document.getElementById('search-district');

    if (!searchInput.contains(event.target) && !searchSuggestions.contains(event.target)) {
        searchSuggestions.classList.remove("active");  // Đóng dropdown khi click ra ngoài
    }
});

// Thêm sự kiện cho nút "Close"
document.getElementById('close-info-tab').addEventListener('click', function() {
    closeInfoTab();  // Gọi hàm đóng tab
});

// Thêm sự kiện focus cho thanh tìm kiếm để đóng tab AQI khi người dùng click vào thanh tìm kiếm
document.querySelector("#search-district").addEventListener('focus', function() {
    closeInfoTab();  // Đóng tab khi người dùng nhấn vào thanh tìm kiếm
});

// Hàm đóng tab AQI
function closeInfoTab() {
    let infoTab = document.getElementById('info-tab');
    infoTab.classList.remove('active');  // Ẩn tab khi nhấn nút close hoặc khi tìm kiếm
    infoTab.style.display = 'none';  // Ẩn tab
}

document.addEventListener('click', function(event) {
    console.log('Bạn đã click vào:', event.target); // Hiển thị phần tử bạn đã click vào
});

const selectHour = document.getElementById('select-hour');
selectHour.addEventListener('click', function() {
    selectHour.style.position = 'absolute';
    selectHour.style.bottom = '0px'; // Luôn xuất hiện ở dưới form container
    selectHour.style.left = '0px';   
});



function getHealthRecommendations(aqi) {
    if (aqi <= 50) {
        return "Air quality is good. No special precautions needed.";
    } else if (aqi <= 100) {
        return "Air quality is moderate. Sensitive individuals should reduce prolonged outdoor activities.";
    } else if (aqi <= 150) {
        return "Unhealthy for sensitive groups. Limit outdoor activities if you have respiratory issues.";
    } else if (aqi <= 200) {
        return "Unhealthy. Reduce outdoor activities and wear a mask if needed.";
    } else if (aqi <= 300) {
        return "Very Unhealthy. Avoid outdoor activities; wear a mask.";
    } else {
        return "Hazardous. Stay indoors, keep windows closed, and use an air purifier if available.";
    }
}

// Biến lưu chỉ số AQI trước đó của các huyện
let previousAQI = {};

// Hàm lấy và kiểm tra sự thay đổi của AQI
function checkAQIChanges() {
    fetch('/get_current_aqi')
        .then(response => response.json())
        .then(currentAQI => {
            for (const district in currentAQI) {
                const newAQI = currentAQI[district];

                // Nếu đã có AQI trước đó của huyện này thì kiểm tra sự thay đổi
                if (previousAQI[district] !== undefined) {
                    const difference = newAQI - previousAQI[district];

                    // Kiểm tra nếu sự thay đổi lớn hơn ±10
                    if (Math.abs(difference) >= 2) {
                        showAQIChangeNotification(district, newAQI, difference);
                    }
                }

                // Cập nhật chỉ số AQI hiện tại vào biến previousAQI
                previousAQI[district] = newAQI;
            }
        })
        .catch(error => console.error('Error fetching AQI data:', error));
}

// Hàm hiển thị thông báo thay đổi AQI dưới dạng toast
function showAQIChangeNotification(district, newAQI, difference) {
    // Tạo phần tử thông báo (toast)
    const notification = document.createElement('div');
    notification.className = 'aqi-toast';
    notification.innerHTML = `
        <strong>AQI Change</strong><br>
        The AQI of ${district} has ${difference > 0 ? 'increased' : 'decreased'} by ${Math.abs(difference)} points, 
        current AQI: ${newAQI}.
`;

    // Đặt style cho toast
    notification.style.position = 'fixed';
    notification.style.bottom = '20px';
    notification.style.right = '20px';
    notification.style.backgroundColor = '#333';
    notification.style.color = '#fff';
    notification.style.padding = '15px';
    notification.style.borderRadius = '5px';
    notification.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.3)';
    notification.style.zIndex = '1000';
    notification.style.opacity = '0';
    notification.style.transition = 'opacity 0.5s ease';

    // Thêm thông báo vào trang
    document.body.appendChild(notification);

    // Hiệu ứng fade-in
    setTimeout(() => {
        notification.style.opacity = '1';
    }, 100);

    // Tự động xóa thông báo sau 5 giây với hiệu ứng fade-out
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => {
            notification.remove();
        }, 500);
    }, 5000);
}

// Gọi hàm checkAQIChanges mỗi 15 phút (900000 ms)
setInterval(checkAQIChanges, 60000); // 15 phút

// Gọi ngay hàm checkAQIChanges lần đầu tiên khi trang được tải
document.addEventListener('DOMContentLoaded', checkAQIChanges);
