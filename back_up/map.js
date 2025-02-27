// Tạo bản đồ với Leaflet.js
var map = L.map('map').setView([10.1, 105.5], 8);  // Tọa độ trung tâm Đồng bằng Sông Cửu Long

// Thêm tile layer từ OpenStreetMap
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18,
}).addTo(map);

// Biến lưu thông tin các huyện
var districtCoords = {};
var districtNames = [];  // Mảng để lưu tên huyện cho autocomplete

// Lấy dữ liệu các huyện từ server
fetch('/get_districts')
    .then(response => response.json())
    .then(districts => {
        // Duyệt qua từng tỉnh và huyện để hiển thị marker trên bản đồ
        for (var province in districts) {
            var districtData = districts[province];
            for (var district in districtData) {
                var coords = districtData[district];
                
                // Lưu tọa độ huyện vào biến districtCoords
                districtCoords[district.toLowerCase()] = { lat: coords.lat, lon: coords.lon };

                // Thêm tên huyện vào mảng gợi ý tìm kiếm
                districtNames.push(district);

                // Thêm marker lên bản đồ
                var marker = L.marker([coords.lat, coords.lon]).addTo(map)
                    .bindPopup(`${district}, ${province}`)
                    .on('click', function(e) {
                        var popupContent = this.getPopup().getContent().split(", ");
                        var districtName = popupContent[0];
                        fetchAQIPrediction(districtName, e);  // Truyền sự kiện click
                    });
            }
        }

        // Khởi tạo autocomplete cho thanh tìm kiếm
        $("#search-district").autocomplete({
            source: districtNames
        });
    });

// Hàm fetch dự đoán AQI cho huyện được chọn và kiểm tra cập nhật dữ liệu
function fetchAQIPrediction(districtName, e) {
    console.log('Fetching AQI data for:', districtName);  // Log để kiểm tra tên huyện gửi đi
    fetch('/get_aqi', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 'district': districtName })
    })
    .then(response => {
        if (response.status === 202) {
            // Nếu dữ liệu cũ, hiển thị thông báo và bắt đầu quá trình cập nhật
            alert('The data is outdated, proceeding to fetch and retrain the model...');
            showProgressBar();  // Hiển thị thanh tiến trình

            // Gọi API để cập nhật dữ liệu và huấn luyện mô hình
            fetch('/update_and_train', { method: 'POST' })
                .then(() => {
                    // Bắt đầu theo dõi quá trình huấn luyện
                    trackTrainingStatus();  // Gọi hàm để liên tục theo dõi trạng thái huấn luyện
                })
                .catch(error => {
                    console.log('Error updating data:', error);
                    hideProgressBar();
                    alert('An error occurred during the update process.');
                });
        } else if (response.ok) {
            return response.json();
        } else {
            return response.json().then(data => { throw new Error(data.error); });
        }
    })
    .then(data => {
        if (data && data.aqi_list) {
            // Chỉ hiển thị `info-tab` nếu dữ liệu đã được cập nhật
            showInfoTab(districtName, e, data);
        }
    })
    .catch(error => {
        console.log('Error:', error);
        document.getElementById('result').innerText = `Error: ${error.message}`;
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

// Hiển thị info-tab khi có dữ liệu mới
function showInfoTab(districtName, e, data) {
    // Cập nhật tiêu đề của tab với tên huyện
    let infoTab = document.getElementById('info-tab');
    infoTab.classList.add('active');
    document.querySelector('.custom-aqi-title').innerText = `AQI results for ${districtName}`;

    // Xóa nội dung cũ trước khi thêm mới
    let tableBody = document.getElementById('aqi-table-body');
    tableBody.innerHTML = '';

    data.aqi_list.forEach((entry, index) => {
        let dateTime = new Date(entry.Timestamp);
        let formattedDate = `${dateTime.getDate()}/${dateTime.getMonth() + 1} ${dateTime.getHours().toString().padStart(2, '0')}:${dateTime.getMinutes().toString().padStart(2, '0')}`;
        
        // Dự đoán mức độ ô nhiễm dựa trên AQI
        let aqiLevel = getAQILevel(entry['AQI Prediction']);
        let aqiColorClass = getAQIColorClass(entry['AQI Prediction']);
        
        // Tạo hàng dữ liệu cho bảng, thêm lớp màu cho hàng dựa trên mức độ ô nhiễm
        let row = `<tr class="${aqiColorClass}" data-aos="fade-up">
            <td>${formattedDate}</td>
            <td>${entry['AQI Prediction'].toFixed(2)} AQI US</td>
            <td>${aqiLevel}</td>
        </tr>`;
        tableBody.innerHTML += row;
    });

    // Cập nhật vị trí tab dựa trên sự kiện click
    infoTab.style.display = 'block';  // Hiển thị tab
    infoTab.style.top = (e.originalEvent.pageY - 100) + 'px';  // Điều chỉnh tọa độ Y
    infoTab.style.left = e.originalEvent.pageX + 'px';  // Điều chỉnh tọa độ X

    // Khởi tạo lại hiệu ứng AOS sau khi dữ liệu mới được thêm
    AOS.refresh();
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
        return 'Tốt';
    } else if (aqi <= 100) {
        return 'Trung bình';
    } else if (aqi <= 150) {
        return 'Không tốt cho người nhạy cảm';
    } else if (aqi <= 200) {
        return 'Không tốt';
    } else if (aqi <= 300) {
        return 'Rất không tốt';
    } else {
        return 'Nguy hiểm';
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

// Xử lý form tìm kiếm và zoom vào huyện với hiệu ứng chuyển động
function searchDistrict(districtName) {
    var nameLower = districtName.toLowerCase();
    if (districtCoords[nameLower]) {
        var coords = districtCoords[nameLower];
        map.flyTo([coords.lat, coords.lon], 13, {
            animate: true,
            duration: 1.5
        });
        fetchAQIPrediction(districtName);  // Gọi hàm fetch AQI khi tìm kiếm
    } else {
        alert("Không tìm thấy huyện " + districtName);
    }
}

// Xử lý form tìm kiếm
document.querySelector("form").addEventListener("submit", function(event) {
    event.preventDefault(); 
    var searchInput = document.querySelector("input[type='search']").value;
    closeInfoTab();  // Đóng tab khi người dùng nhấn tìm kiếm
    searchDistrict(searchInput);
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
