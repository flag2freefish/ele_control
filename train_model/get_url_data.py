
import get_data_update as get_data
import matplotlib.pyplot as plt

begin = '2023-05-22 00:00:00'
end = '2023-06-04 23:59:59'
mydata = get_data.get_load_A_park(begin, end)
pvdata = get_data.get_power_pvA(begin, end)
mydata = get_data.get_load_B_park(begin, end)

print(len(mydata))
# print(mydata)

begin1 = '2023-06-01 00:00:00'
end1 = '2023-06-01 23:59:59'
mydata1 = get_data.get_userdb(begin, end, '1h', 'eqm000000', 'opt010011')
print(mydata1)
plt.plot(pvdata, label='real')
plt.plot(mydata1, label='pred')
plt.legend()
plt.show()
